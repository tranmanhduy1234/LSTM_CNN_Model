import numpy as np
import sounddevice as sd
import time

# Cache for throttle control so sound doesn't overlap excessively
_last_sound_time = 0.0

def play_beep(frequency=1000, duration=0.3, volume=0.3):
    """
    Generate and play a synthetic beep tone using numpy and sounddevice.
    This runs asynchronously and does not block the video display loop.
    """
    global _last_sound_time
    now = time.time()
    
    # Throttle beep triggers so they don't overlap too closely (max one beep per 0.4 seconds)
    if now - _last_sound_time < (duration + 0.1):
        return
        
    _last_sound_time = now
    
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Generate sine wave
        wave = np.sin(frequency * t * 2 * np.pi) * volume
        # Ensure smooth decay at the end to prevent clicking noises
        fade_out_len = int(sample_rate * 0.05)
        fade_out = np.linspace(1.0, 0.0, fade_out_len)
        wave[-fade_out_len:] *= fade_out
        
        # Play asynchronously
        sd.play(wave, sample_rate)
    except Exception as e:
        # Fallback if audio hardware is busy or not available
        print(f"[AUDIO RUNTIME ERROR] Cannot play sound: {e}")

def trigger_alerts(drowsiness_prob, threshold=0.5):
    """
    Evaluates risk probability and outputs multi-modal alarms:
      - Auditory: Variable frequency tone based on risk level.
      - Visual/Console: Prints physical warning light signals (simulation of serial/GPIO triggers).
    """
    # Level 1: Awake (Normal)
    if drowsiness_prob < 0.4:
        # Mock signal to turn off hardware lights or set to GREEN
        # print("[HARDWARE TRIGGER] LED = GREEN, BUZZER = OFF")
        pass
        
    # Level 2: Mild Fatigue (Warning)
    elif drowsiness_prob < threshold:
        # Yellow Warning Alert
        # Beep with low frequency / gentle pattern
        play_beep(frequency=600, duration=0.15, volume=0.15)
        print("[HARDWARE TRIGGER] LED = YELLOW (Warning), BUZZER = OFF")
        
    # Level 3: Severe Drowsiness (Danger)
    else:
        # Urgent Red Alarm
        # High pitch rapid beep pattern
        play_beep(frequency=1200, duration=0.4, volume=0.5)
        print("[HARDWARE TRIGGER] LED = FLASHING RED (Danger), BUZZER = ON (Active)")
