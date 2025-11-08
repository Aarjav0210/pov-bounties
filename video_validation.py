import torch
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import ruptures as rpt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2-VL-7B-Instruct"

def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=512*28*28)
    return model, processor

def extract_frames(video_path: Path, fps: float = 0.2) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    num_frames = int(duration * fps)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            frames.append(pil_image)

    cap.release()
    return frames

def extract_scene_frames(video_path, start_sec, end_sec, max_frames=12):
    """Extract frames from a specific scene for Qwen."""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate start and end frame indices
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    total_scene_frames = end_frame - start_frame + 1

    # Determine number of frames to extract
    # Ensure num_frames is at least 1 and not more than total_scene_frames
    num_frames = min(max_frames, total_scene_frames)
    if num_frames <= 0:
        return []

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            # Resize to reduce memory, consistent with Qwen usage
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            frames.append(pil_image)

    cap.release()
    return frames

def extract_features(video_path: Path, feature_type: str = "color_hist") -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Extracting features from {total_frames} frames...")

    features = []
    prev_frame = None

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        if feature_type == 'color_hist':
            # Color histogram - good for detecting scene/lighting changes
            frame_small = cv2.resize(frame, (160, 90))
            hist = []
            for i in range(3):  # B, G, R channels
                h = cv2.calcHist([frame_small], [i], None, [32], [0, 256])
                hist.extend(h.flatten())
            features.append(hist)

        elif feature_type == 'frame_diff':
            # Frame difference - good for detecting motion/action changes
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_small = cv2.resize(frame_gray, (160, 90))

            if prev_frame is not None:
                diff = cv2.absdiff(frame_small, prev_frame)
                feat = [diff.mean(), diff.std(), np.median(diff)]
                features.append(feat)
            else:
                features.append([0, 0, 0])

            prev_frame = frame_small

        elif feature_type == 'combined':
            # Combined features - best for task detection
            frame_small = cv2.resize(frame, (160, 90))
            frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            # Color stats
            color_mean = frame_small.mean(axis=(0, 1))
            color_std = frame_small.std(axis=(0, 1))

            # Motion
            if prev_frame is not None:
                diff = cv2.absdiff(frame_gray, prev_frame)
                motion_mean = diff.mean()
                motion_std = diff.std()
            else:
                motion_mean = 0
                motion_std = 0

            feat = [*color_mean, *color_std, motion_mean, motion_std]
            features.append(feat)

            prev_frame = frame_gray

    cap.release()

    signal = np.array(features)
    print(f"✓ Extracted features: {signal.shape}")

    return signal, fps, total_frames

def detect_task_changes(video_path: Path, method: str = 'pelt', pen: float = 10.0, feature_type: str = 'combined') -> list[int]:
    # Extract features
    signal, fps, total_frames = extract_features(video_path, feature_type)

    # Detect change points
    print(f"\n🔍 Detecting change points (method={method}, penalty={pen})...")

    if method == 'pelt':
        # PELT - fastest, automatically determines number of change points
        algo = rpt.Pelt(model="rbf", min_size=int(fps*2), jump=5).fit(signal)
        change_frames = algo.predict(pen=pen)

    elif method == 'window':
        # Window-based - good for local changes
        algo = rpt.Window(width=int(fps*5), model="l2").fit(signal)
        change_frames = algo.predict(n_bkps=10)  # Predict ~10 changes

    elif method == 'binseg':
        # Binary segmentation - balanced approach
        algo = rpt.Binseg(model="l2", min_size=int(fps*2), jump=5).fit(signal)
        change_frames = algo.predict(n_bkps=10)

    # Convert to scenes with timestamps
    scenes = []
    start_frame = 0

    for i, end_frame in enumerate(change_frames, 1):
        start_sec = start_frame / fps
        end_sec = end_frame / fps

        scenes.append({
            'scene_num': i,
            'start': start_sec,
            'end': end_sec,
            'duration': end_sec - start_sec,
            'start_frame': start_frame,
            'end_frame': end_frame,
        })

        start_frame = end_frame

    print(f"✓ Detected {len(scenes)} task changes\n")

    return scenes

def analyze_scene(frames, question, processor, model, max_tokens=200):
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": frame} for frame in frames],
            {"type": "text", "text": question},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response

def analyze_scene_with_smart_retry(frames, initial_question, processor, model, max_retries=3, max_tokens=200):
    attempts = []

    for attempt_num in range(1, max_retries + 1):
        print(f"      Attempt {attempt_num}/{max_retries}...")

        # Progressive prompting
        if attempt_num == 1:
            question = initial_question
        elif attempt_num == 2:
            prev_desc = attempts[-1]['description']
            prev_verify = attempts[-1]['verification']
            question = f"{initial_question}\n\nPrevious answer '{prev_desc}' was incorrect because: {prev_verify}\n\nLook more carefully and describe ONLY what you can clearly observe."
        else:
            # Show all previous attempts
            prev_attempts = "\n".join([f"- Attempt {a['attempt']}: {a['description']} (rejected)"
                                       for a in attempts])
            question = f"{initial_question}\n\nPrevious attempts were all rejected:\n{prev_attempts}\n\nBe very specific and accurate. Describe only visible actions."

        description = analyze_scene(frames, question, processor, model, max_tokens=200)

        # Verification
        verify_q = f"Is this statement true: '{description}'? Justify your answer."
        # verify_q = f"What action(s) are happening in this video?"
        verification = analyze_scene(frames, verify_q, processor, model, max_tokens=200)

        # Parse verification
        verify_lower = verification.lower().strip()
        is_verified = (verify_lower.startswith('yes') or
                      verify_lower.startswith('yes,') or
                      verify_lower.startswith('yes.'))

        attempts.append({
            'attempt': attempt_num,
            'description': description,
            'verification': verification,
            'verified': is_verified
        })

        print(f"         {description}")
        print(f"         Verify: {'✅ Yes' if is_verified else '❌ No'}")
        print(f"         Verification Output: {verify_lower}")

        if is_verified:
            return {
                'description': description,
                'verified': True,
                'attempts': attempt_num,
                'verification_history': attempts,
                'final_verification': verification
            }

    return {
        'description': attempts[-1]['description'],
        'verified': False,
        'attempts': max_retries,
        'verification_history': attempts,
        'final_verification': attempts[-1]['verification']
    }

def analyze_scenes_with_retry_loop(video_path, scenes, question_template, processor, model,
                                   retry_function=analyze_scene_with_smart_retry, max_retries=3, max_frames=12):
    verified_results = []
    failed_results = []
    previous_description = None

    for scene in scenes:
        scene_num = scene['scene_num']

        print(f"\n🎬 Scene {scene_num}: {scene['start']:.1f}s - {scene['end']:.1f}s")

        # Extract frames
        frames = extract_scene_frames(video_path, scene['start'], scene['end'], max_frames)
        print(f"   Extracted {len(frames)} frames")

        # Build question with context
        if previous_description:
            question = f"{question_template}\n\nPrevious scene: {previous_description}"
        else:
            question = question_template

        # Analyze with verification
        print(f"   🔄 Analyzing with self-verification (max {max_retries} attempts)...")
        analysis = retry_function(frames, question, processor, model, max_retries, max_tokens=200)

        scene_result = {
            **scene,
            **analysis,
            'previous_context': previous_description
        }

        # Categorize result
        if analysis['verified']:
            verified_results.append(scene_result)
            previous_description = analysis['description']
            print(f"   ✅ Added to verified list")
        else:
            failed_results.append(scene_result)
            print(f"   ⚠️  Added to failed list (unverified)")

    return verified_results, failed_results


# Cell: Text-only LLM validation

def validate_task_with_llm(verified_results, expected_task, processor, model):
    # Compile all scene descriptions
    scene_descriptions = []
    for i, result in enumerate(verified_results, 1):
        scene_descriptions.append(
            f"Scene {result['scene_num']} ({result['start']:.1f}s-{result['end']:.1f}s): {result['description']}"
        )

    full_description = "\n".join(scene_descriptions)

    # Create validation prompt
    validation_prompt = f"""I have analyzed a video and identified the following sequence of actions:

{full_description}

Based on these scene descriptions, is this video showing the task: "{expected_task}"?

Answer in this format:
1. VERDICT: Yes or No
2. CONFIDENCE: High, Medium, or Low
3. REASONING: Explain why, citing specific scenes
4. MISSING ELEMENTS: What's missing if it's not a complete match
"""

    print("🤖 Validating with LLM...")
    print(f"Expected task: {expected_task}\n")

    # Use Qwen2-VL in text-only mode
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": validation_prompt}],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=400)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    validation = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Parse verdict
    validation_lower = validation.lower()
    verdict_confirmed = 'verdict: yes' in validation_lower or \
                       'verdict:yes' in validation_lower or \
                       validation_lower.strip().startswith('1. verdict: yes')

    return {
        'expected_task': expected_task,
        'num_scenes': len(verified_results),
        'scene_descriptions': scene_descriptions,
        'validation': validation,
        'confirmed': verdict_confirmed
    }


def main():
    model, processor = load_model()
    video_path = Path("chopping-tomatoes.mp4")
    scenes = detect_task_changes(video_path, method='pelt', pen=8)
    question="What action(s) is happening in this scene? Be specific, not vague."
    verified_results, failed_results = analyze_scenes_with_retry_loop(
        video_path = video_path,
        scenes = scenes,
        question_template = question,
        retry_function = analyze_scene_with_smart_retry,
        processor = processor,
        model = model,
        max_retries=3,
        max_frames=12
    )

    expected_task = "chopping tomatoes"

    validation_result = validate_task_with_llm(
        verified_results,
        expected_task=expected_task,
        processor=processor,
        model=model
    )

if __name__ == "__main__":
    main()