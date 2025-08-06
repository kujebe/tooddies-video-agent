from PIL import Image, ImageDraw, ImageFont
import os
import string
from kokoro import KPipeline
import soundfile as sf
from typing import List, Dict
import subprocess
import numpy as np
import re

CUDA = os.environ.get('CUDA', '0') == '1'

def _render_dynamic_red_box(
    overlay,  # The main Image object to paste onto
    draw,  # The ImageDraw object for the overlay
    overlay_total_width, # Total width of the main overlay image
    box_y_start,  # Y-coordinate for the top of the box
    box_height_abs,  # Absolute height of the box
    display_name,  # Text to display in the box
    font_path,  # Path to the .ttf font file
    volume_icon_path,  # Path to the volume icon image
    base_color_rgb  # Tuple (R, G, B) for the box background color
):
    """
    Renders a dynamically sized box with a solid part, a gradient part,
    an icon, and text.
    """
    # A) Text dimensions for the display name in the box
    font_size = int(box_height_abs * 0.6)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = font.getbbox(display_name)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1] # Full height of text
    text_y_offset = text_bbox[1] # Y offset for precise vertical centering
    
    print(f"Text dimensions: {text_width}x{text_height}, Y offset: {text_y_offset}") 

    # B) Volume icon dimensions for the box
    vol_icon_image = Image.open(volume_icon_path).convert("RGBA")
    
    icon_height = int(box_height_abs * 0.7) # Icon height 70% of box
    _icon_aspect_ratio = vol_icon_image.width / vol_icon_image.height
    icon_width = int(icon_height * _icon_aspect_ratio)
    vol_icon_resized = vol_icon_image.resize((icon_width, icon_height))

    # C) Define the "5% padding segment" width
    _padding_reference_width = int(overlay_total_width * 0.25) 
    padding_segment_amount = int(_padding_reference_width * 0.08)

    # D) Calculate the new dynamic box_width
    # Layout: [P] Icon [P] Text [P][P][P]  (P = padding_segment_amount)
    dynamic_box_width = (padding_segment_amount * 2) + icon_width + text_width + (padding_segment_amount * 7)
    dynamic_box_width = max(dynamic_box_width, padding_segment_amount * 5) 

    # Create the solid part (e.g., 60% of the box width for a 40% gradient)
    solid_width = int(dynamic_box_width * 0.7)
    draw.rectangle(
        [(0, box_y_start), (solid_width, box_y_start + box_height_abs)],
        fill=(*base_color_rgb, 255)  # Solid color with full opacity
    )
    
    # Create the gradient part (now 40% of the box width)
    gradient_width = dynamic_box_width - solid_width
    
    if gradient_width > 0:
        gradient_img = Image.new("RGBA", (gradient_width, box_height_abs+1), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient_img)
        for x_grad in range(gradient_width):
            alpha = int(255 * (1 - (x_grad / gradient_width)))
            gradient_draw.line([(x_grad, 0), (x_grad, box_height_abs+100)], fill=(*base_color_rgb, alpha))
        overlay.paste(gradient_img, (solid_width, box_y_start), gradient_img)

    # Position volume icon and text within the dynamically-sized box
    icon_x_position = padding_segment_amount
    icon_y_position = int(box_y_start + (box_height_abs - icon_height) / 2)
    overlay.paste(vol_icon_resized, (icon_x_position, icon_y_position), vol_icon_resized)

    name_x_position = icon_x_position + icon_width + padding_segment_amount
    name_y_position = int(box_y_start + (box_height_abs - text_height) / 2 - text_y_offset)
    draw.text((name_x_position, name_y_position), display_name, font=font, fill="white")
    
    return solid_width

def create_overlay(
    person_image_path,
    volume_icon_path,
    display_name,
    output_path,
    base_color_rgb=(232, 14, 64),  # Original red color
    subtitle_background_color=(0, 0, 0, 99),  # Semi-transparent black
    font_path="assets/noto.ttf",  # Make sure to have a suitable .ttf font file available
    overlay_size=(1920, 1080)  # Set to 1920x1080 as per requirements
):
    # Base overlay image (transparent)
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Calculate sizes and positions based on percentages
    width, height = overlay_size
    
    # Define layout percentages
    top_margin_percent = 0.10
    person_image_height_percent = 0.40
    red_box_height_percent = 0.10
    black_box_height_percent = 0.30
    # Bottom margin is implicitly 1.0 - sum of above = 0.10

    # Calculate absolute pixel values for heights
    top_margin_abs = int(height * top_margin_percent)
    person_image_height_abs = int(height * person_image_height_percent)
    red_box_height_abs = int(height * red_box_height_percent)
    black_box_height_abs = int(height * black_box_height_percent)

    # Calculate Y positions
    person_y_start = top_margin_abs
    red_box_y_start = person_y_start + person_image_height_abs
    box_y_start = red_box_y_start + red_box_height_abs # This is the semi-transparent black box
    
    # 1. Semi-transparent black box - NOW USING THE PARAMETER VALUE
    draw.rectangle(
        [(0, box_y_start), (width, box_y_start + black_box_height_abs)],
        fill=subtitle_background_color  # Using the parameter directly
    )
    
    # Call the new helper function to render the red box
    solid_width = _render_dynamic_red_box(
        overlay=overlay,
        draw=draw,
        overlay_total_width=width,
        box_y_start=red_box_y_start,
        box_height_abs=red_box_height_abs,
        display_name=display_name,
        font_path=font_path,
        volume_icon_path=volume_icon_path,
        base_color_rgb=base_color_rgb
    )

    # 3. Position the person image
    person_img = Image.open(person_image_path).convert("RGBA")
    person_img_original_width, person_img_original_height = person_img.size
    person_width_abs = int(person_image_height_abs * person_img_original_width / person_img_original_height)
    
    person_x_start = 15
    
    # Resize and paste person image
    person_img_resized = person_img.resize((person_width_abs, person_image_height_abs))
    overlay.paste(person_img_resized, (person_x_start, person_y_start), person_img_resized)
    
    # Save the overlay
    overlay.save(output_path)
    print(f"Overlay saved to {output_path}")

def create_tts_english(text, output_path, lang_code, voice):
    pipeline = KPipeline(
        lang_code=lang_code,
    )

    generator = pipeline(text, voice=voice)

    captions = []
    audio_data = []
    full_audio_length = 0
    for i, result in enumerate(generator):
        data = result.audio
        audio_length = len(data) / 24000
        audio_data.append(data)
        if result.tokens:
            tokens = result.tokens
            for t in tokens:
                if t.start_ts is None or t.end_ts is None:
                    if captions:
                        captions[-1]["text"] += t.text
                        captions[-1]["end_ts"] = full_audio_length + audio_length
                    continue
                try:
                    captions.append({
                        "text": t.text,
                        "start_ts": full_audio_length + t.start_ts,
                        "end_ts": full_audio_length + t.end_ts
                    })
                except Exception as e:
                    print(f"Processing error in Kokoro, there's a character that is not supported. text: {t.text}")
                    raise ValueError(f"Error processing token: {t}, Error: {e}")
        full_audio_length += audio_length
    
    audio_data = np.concatenate(audio_data)
    sf.write(output_path, audio_data, 24000)
    return captions, full_audio_length

def break_text_into_sentences(text, lang_code) -> List[str]:
    """
    Advanced sentence splitting with better handling of abbreviations and edge cases.
    """
    if not text or not text.strip():
        return []
    
    # Language-specific sentence boundary patterns
    patterns = {
        'a': r'(?<=[.!?])\s+(?=[A-Z])',     # English
        'e': r'(?<=[.!?¿¡])\s+(?=[A-ZÁÉÍÓÚÑÜ])',  # Spanish
        'f': r'(?<=[.!?])\s+(?=[A-ZÁÀÂÄÇÉÈÊËÏÎÔÖÙÛÜŸ])',  # French
        'h': r'(?<=[।!?])\s+',           # Hindi: Split after devanagari danda
        'i': r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß])',  # Italian
        'p': r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ])',  # Portuguese
        'z': r'(?<=[。！？])',            # Chinese: Split after Chinese punctuation
    }
    
    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = {
        'e': {'Sr.', 'Sra.', 'Dr.', 'Dra.', 'Prof.', 'etc.', 'pág.', 'art.', 'núm.', 'cap.', 'vol.'},  # Spanish
        'f': {'M.', 'Mme.', 'Dr.', 'Prof.', 'etc.', 'art.', 'p.', 'vol.', 'ch.', 'fig.', 'n°'},  # French
        'h': {'श्री', 'श्रीमती', 'डॉ.', 'प्रो.', 'etc.', 'पृ.', 'अध.'},  # Hindi
        'i': {'Sig.', 'Sig.ra', 'Dr.', 'Prof.', 'ecc.', 'pag.', 'art.', 'n.', 'vol.', 'cap.', 'fig.'},  # Italian
        'p': {'Sr.', 'Sra.', 'Dr.', 'Dra.', 'Prof.', 'etc.', 'pág.', 'art.', 'n.º', 'vol.', 'cap.'},  # Portuguese
        'z': {'先生', '女士', '博士', '教授', '等等', '第', '页', '章'}  # Chinese
    }
    
    abbrevs = abbreviations.get(lang_code, set())
    
    # Protect abbreviations by temporarily replacing them
    protected_text = text
    replacements = {}
    for i, abbrev in enumerate(abbrevs):
        placeholder = f"__ABBREV_{i}__"
        protected_text = protected_text.replace(abbrev, placeholder)
        replacements[placeholder] = abbrev
    
    # Apply the regex splitting
    pattern = patterns.get(lang_code, patterns['a'])
    sentences = re.split(pattern, protected_text.strip())
    
    # Restore abbreviations and clean up
    restored_sentences = []
    for sentence in sentences:
        for placeholder, original in replacements.items():
            sentence = sentence.replace(placeholder, original)
        sentence = sentence.strip()
        if sentence:
            restored_sentences.append(sentence)
    
    return restored_sentences if restored_sentences else [text.strip()]

def create_tts_international(text, output_path, lang_code, voice):
    sentences = break_text_into_sentences(text, lang_code)

    # generate the audio for each sentence
    audio_data = []
    captions = []
    full_audio_length = 0
    pipeline = KPipeline(
        lang_code=lang_code,
    )
    for sentence in sentences:
        generator = pipeline(sentence, voice=voice)
        
        for i, result in enumerate(generator):
            data = result.audio
            audio_length = len(data) / 24000
            audio_data.append(data)
            # since there are no tokens, we can just use the sentence as the text
            captions.append({
                "text": sentence,
                "start_ts": full_audio_length,
                "end_ts": full_audio_length + audio_length
            })
            full_audio_length += audio_length
    audio_data = np.concatenate(audio_data)
    sf.write(output_path, audio_data, 24000)
    return captions, full_audio_length

def is_punctuation(text):
    return text in string.punctuation

def create_subtitle_segments_english(captions: List[Dict], max_length=80, lines=2):
    """
    Breaks up the captions into segments of max_length characters
    on two lines and merge punctuation with the last word
    """
    
    if not captions:
        return []
    
    segments = []
    current_segment_texts = ["" for _ in range(lines)]
    current_line = 0
    segment_start_ts = captions[0]["start_ts"]
    segment_end_ts = captions[0]["end_ts"]
    
    for caption in captions:
        text = caption["text"]
        start_ts = caption["start_ts"]
        end_ts = caption["end_ts"]
        
        # Update the segment end timestamp
        segment_end_ts = end_ts
        
        # If the caption is a punctuation, merge it with the current line
        if is_punctuation(text):
            if current_line < lines and current_segment_texts[current_line]:
                current_segment_texts[current_line] += text
            continue
        
        # If the line is too long, move to the next one
        if current_line < lines and len(current_segment_texts[current_line] + text) > max_length:
            current_line += 1
        
        # If we've filled all lines, save the current segment and start a new one
        if current_line >= lines:
            segments.append({
                "text": current_segment_texts,
                "start_ts": segment_start_ts,
                "end_ts": segment_end_ts
            })
            
            # Reset for next segment
            current_segment_texts = ["" for _ in range(lines)]
            current_line = 0
            # Add a small gap (0.05s) between segments to prevent overlap
            segment_start_ts = start_ts + 0.05
        
        # Add the text to the current segment
        if current_line < lines:
            current_segment_texts[current_line] += " " if current_segment_texts[current_line] else ""
            current_segment_texts[current_line] += text
    
    # Add the last segment if there's any content
    if any(current_segment_texts):
        segments.append({
            "text": current_segment_texts,
            "start_ts": segment_start_ts,
            "end_ts": segment_end_ts
        })
    
    # Post-processing to ensure no overlaps by adjusting end times if needed
    for i in range(len(segments) - 1):
        if segments[i]["end_ts"] >= segments[i+1]["start_ts"]:
            segments[i]["end_ts"] = segments[i+1]["start_ts"] - 0.05
    
    return segments 

def create_subtitle_segments_international(captions: List[Dict], max_length=80, lines=2):
    """
    Breaks up international captions (full sentences) into smaller segments that fit
    within max_length characters per line, with proper timing distribution.
    
    Handles both space-delimited languages like English and character-based languages like Chinese.
    
    Args:
        captions: List of caption dictionaries with text, start_ts, and end_ts
        max_length: Maximum number of characters per line
        lines: Number of lines per segment
        
    Returns:
        List of subtitle segments
    """
    if not captions:
        return []
    
    segments = []
    
    for caption in captions:
        text = caption["text"].strip()
        start_ts = caption["start_ts"]
        end_ts = caption["end_ts"]
        duration = end_ts - start_ts
        
        # Check if text is using Chinese/Japanese/Korean characters (CJK)
        # For CJK, we'll split by characters rather than words
        is_cjk = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        parts = []
        if is_cjk:
            # For CJK languages, process character by character
            current_part = ""
            for char in text:
                if len(current_part + char) > max_length:
                    parts.append(current_part)
                    current_part = char
                else:
                    current_part += char
            
            # Add the last part if not empty
            if current_part:
                parts.append(current_part)
        else:
            # Original word-based splitting for languages with spaces
            words = text.split()
            current_part = ""
            
            for word in words:
                # If adding this word would exceed max_length, start a new part
                if len(current_part + " " + word) > max_length and current_part:
                    parts.append(current_part.strip())
                    current_part = word
                else:
                    # Add space if not the first word in the part
                    if current_part:
                        current_part += " "
                    current_part += word
            
            # Add the last part if not empty
            if current_part:
                parts.append(current_part.strip())
        
        # Group parts into segments with 'lines' number of lines per segment
        segment_parts = []
        for i in range(0, len(parts), lines):
            segment_parts.append(parts[i:i+lines])
        
        # Calculate time proportionally based on segment text length
        total_chars = sum(len(''.join(part_group)) for part_group in segment_parts)
        
        current_time = start_ts
        for i, part_group in enumerate(segment_parts):
            # Get character count for this segment group
            segment_chars = len(''.join(part_group))
            
            # Calculate time proportionally, but ensure at least a minimum duration
            if total_chars > 0:
                segment_duration = (segment_chars / total_chars) * duration
                segment_duration = max(segment_duration, 0.5)  # Ensure minimum duration of 0.5s
            else:
                segment_duration = duration / len(segment_parts)
            
            segment_start = current_time
            segment_end = segment_start + segment_duration
            
            # Move current time forward for next segment
            current_time = segment_end
            
            # Create segment with proper text array format for the subtitle renderer
            segment_text = part_group + [""] * (lines - len(part_group))
            
            segments.append({
                "text": segment_text,
                "start_ts": segment_start,
                "end_ts": segment_end
            })
    
    # Ensure no overlaps between segments by adjusting end times if needed
    for i in range(len(segments) - 1):
        if segments[i]["end_ts"] >= segments[i+1]["start_ts"]:
            segments[i]["end_ts"] = segments[i+1]["start_ts"] - 0.05
    
    return segments

def create_subtitle(segments, output_path, font_size=24, font_color="&H00FFFFFF"):
    # Create the .ass subtitle file with headers
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{font_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,0,0,8,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(font_size=font_size, font_color=font_color)
    
    # Position at 67.5% from the top is achieved using alignment 8 (center) and \pos tag
    pos_y = int(1080 * 0.675)
    
    # Process each segment and add to the subtitle file
    for segment in segments:
        start_time = format_time(segment["start_ts"])
        end_time = format_time(segment["end_ts"])
        
        # Create text with line breaks
        text_lines = segment["text"]
        formatted_text = ""
        for i, line in enumerate(text_lines):
            if line:  # Only add non-empty lines
                if i > 0:  # Add line break if not the first line
                    formatted_text += "\\N"
                formatted_text += line
        
        # Position at 65% from top, horizontally centered
        formatted_text = f"{{\\pos(960,{pos_y})}}" + formatted_text
        
        # Add the dialogue line to the subtitle content
        ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted_text}\n"
    
    # Write the subtitle file
    output_file = f"{output_path}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    print(f"Subtitle file created: {output_file}")
    return output_file

def format_time(seconds):
    """
    Convert seconds to ASS time format (H:MM:SS.cc)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def render_video(sound_path, subtitle_path, overlay_path, audio_length, bg_video_path, output_path):
    """
    Renders a video with the given sound, subtitle, overlay image, and a background video.
    Captures and processes ffmpeg output to track progress.
    
    Args:
        sound_path (str): Path to the audio file
        subtitle_path (str): Path to the ASS subtitle file
        overlay_path (str): Path to the overlay PNG image
        audio_length (float): Length of the audio in seconds
        bg_video_path (str): Path to the background video that will loop
        output_path (str): Path where the output video will be saved
        
    Returns:
        boolean: success of the video creation
    """
    
    try:
        cmd = [
            'ffmpeg', '-y'
        ]
        
        if CUDA:
            cmd.extend(['-hwaccel', 'cuda'])
        
        cmd.extend([
            '-stream_loop', '-1',
            '-t', str(audio_length),
            '-i', bg_video_path,
            '-i', overlay_path,
            '-i', sound_path,
            '-filter_complex', f"[0:v]scale=1920:1080[scaled];[scaled][1:v]overlay=format=auto[overlaid];[overlaid]subtitles={subtitle_path}[v]",
            '-map', '[v]',
            '-map', '2:a',
            '-c:v',  'h264_nvenc' if CUDA else 'libx264',
            '-preset', 'fast' if CUDA else 'ultrafast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-t', str(audio_length),
            output_path
        ])
        
        # Use Popen instead of run to capture output in real-time
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            text=True
        )
        
        # Process the output line by line as it becomes available
        for line in process.stderr:
            # Extract time information for progress tracking
            if "time=" in line and "speed=" in line:
                try:
                    # Extract the time information
                    time_str = line.split("time=")[1].split(" ")[0]
                    # Convert HH:MM:SS.MS format to seconds
                    h, m, s = time_str.split(":")
                    seconds = float(h) * 3600 + float(m) * 60 + float(s)
                    
                    # Calculate progress percentage
                    progress = min(100, (seconds / audio_length) * 100)
                    print(f"Processing: {progress:.2f}% complete (Time: {time_str} / Total: {format_time(audio_length)})")
                except (ValueError, IndexError) as e:
                    # If parsing fails, just print the line
                    print(f"ffmpeg: {line.strip()}")
            elif any(keyword in line for keyword in [
                # Skip initialization information
                "ffmpeg version", "built with", "configuration:", "libav",
                "Input #", "Metadata:", "Duration:", "Stream #",
                "Press [q]", "Output #", "Stream mapping:",
                
                # Skip processing details
                "frame=", "fps=", 
                "[libx264", "kb/s:", "Qavg:", 
                "video:", "audio:", "subtitle:", 
                "frame I:", "frame P:", "mb I", "mb P",
                "coded y,", "i16 v,h,dc,p:", "i8c dc,h,v,p:",
                "compatible_brands:", "encoder", "Side data:",
                "libswscale", "libswresample", "libpostproc",
                
                # Fix the missing commas in this line
                "ffmpeg: libswscale", "ffmpeg: libswresample", "ffmpeg: libpostproc"
            ]):
                # Skip all technical output lines
                pass
            else:
                # Only print important messages (like errors and warnings)
                # that don't match any of the filtered patterns
                if not line.strip() or line.strip().startswith('['):
                    continue
                
                # Skip header lines that describe inputs
                if ":" in line and any(header in line for header in 
                                      ["major_brand", "minor_version", "creation_time", 
                                       "handler_name", "vendor_id", "Duration", "bitrate"]):
                    continue
                
                print(f"ffmpeg: {line.strip()}")
        
        # Wait for the process to complete and check the return code
        return_code = process.wait()
        if return_code != 0:
            print(f"ffmpeg exited with code: {return_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during video rendering: {e}")
        return False
