#!/usr/bin/env python3
"""
Audio Transcription Script using OpenAI Whisper
This script handles the ffmpeg dependency and transcribes audio files.
"""

import os
import sys
import whisper
import subprocess
from pathlib import Path

def setup_ffmpeg():
    """Add ffmpeg to PATH if it's in the project directory"""
    current_dir = Path(__file__).parent
    ffmpeg_dir = current_dir / "ffmpeg-master-latest-win64-gpl" / "bin"
    
    if ffmpeg_dir.exists():
        ffmpeg_path = str(ffmpeg_dir)
        if ffmpeg_path not in os.environ['PATH']:
            os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
            print(f"Added ffmpeg to PATH: {ffmpeg_path}")
        return True
    else:
        print("FFmpeg not found in project directory. Please install ffmpeg.")
        return False

def transcribe_audio(audio_path, model_size="base", language=None):
    """
    Transcribe audio using OpenAI's Whisper model
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
        language (str): Language code (optional)
    
    Returns:
        dict: Transcription result
    """
    try:
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        
        print(f"Transcribing: {audio_path}")
        
        # Transcribe with optional language specification
        if language:
            result = model.transcribe(audio_path, language=language)
        else:
            result = model.transcribe(audio_path)
        
        return result
        
    except Exception as e:
        print(f"Error with {model_size} model: {e}")
        print(f"Error type: {type(e).__name__}")
        return None

def main():
    """Main function to run transcription"""
    # Setup ffmpeg
    if not setup_ffmpeg():
        return
    
    # Find audio files in current directory
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path('.').glob(f'*{ext}'))
    
    if not audio_files:
        print("No audio files found in current directory")
        return
    
    print(f"Found audio files: {[f.name for f in audio_files]}")
    print("=" * 50)
    
    # Test with base model
    print("Testing Whisper BASE model")
    print("=" * 50)
    
    # Use the first audio file found
    audio_file = str(audio_files[0])
    print(f"Using audio file: {audio_file}")
    print(f"File exists: {os.path.exists(audio_file)}")
    print(f"File size: {os.path.getsize(audio_file)} bytes")
    
    # Transcribe
    result = transcribe_audio(audio_file, model_size="base")
    
    if result:
        print("\n" + "=" * 50)
        print("TRANSCRIPTION SUCCESSFUL!")
        print("=" * 50)
        print(f"Text: {result['text']}")
        print(f"Language: {result.get('language', 'Unknown')}")
        print(f"Segments: {len(result.get('segments', []))}")
        
        # Save to file
        output_file = f"whisper_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WHISPER TRANSCRIPTION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Audio file: {audio_file}\n")
            f.write(f"Model: base\n")
            f.write(f"Language: {result.get('language', 'Unknown')}\n")
            f.write(f"Text: {result['text']}\n")
            
            if 'segments' in result:
                f.write("\nDetailed segments:\n")
                for i, segment in enumerate(result['segments']):
                    f.write(f"\nSegment {i+1}:\n")
                    f.write(f"  Start: {segment['start']:.2f}s\n")
                    f.write(f"  End: {segment['end']:.2f}s\n")
                    f.write(f"  Text: {segment['text']}\n")
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("Transcription failed!")

if __name__ == "__main__":
    main() 