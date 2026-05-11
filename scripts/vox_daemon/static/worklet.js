// AudioWorklet processor that forwards 16 kHz mono PCM frames to the main
// thread. WhisperLiveKit (launched with --pcm-input) reinterprets the bytes
// as int16-LE PCM (see venv/.../whisperlivekit/audio_processor.py:209
// `np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0`),
// so we convert Float32 [-1, 1] -> Int16 here before sending. Each
// process() callback delivers 128 samples = 8 ms at 16 kHz; 125 msgs/sec is
// well within localhost WebSocket headroom.

class PCMStreamer extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      const src = input[0];
      const out = new Int16Array(src.length);
      for (let i = 0; i < src.length; i++) {
        const clipped = Math.max(-1, Math.min(1, src[i]));
        out[i] = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
      }
      this.port.postMessage(out.buffer, [out.buffer]);
    }
    return true;  // Keep the processor alive for the duration of the stream
  }
}

registerProcessor('pcm-streamer', PCMStreamer);
