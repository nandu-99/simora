const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs-extra');
const { pipeline } = require('@xenova/transformers');
const ffmpeg = require('fluent-ffmpeg');
ffmpeg.setFfmpegPath(require('ffmpeg-static'));
const WavDecoder = require('wav-decoder');

/**
 * Convert MP3 to 16kHz mono WAV
 * @param {string} inputPath - Path to MP3 file
 * @returns {Promise<string>} - Path to temporary WAV file
 */
async function convertToWav(inputPath) {
  const wavPath = path.join(path.dirname(inputPath), `${path.basename(inputPath, path.extname(inputPath))}.wav`);
  
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .audioChannels(1)
      .audioFrequency(16000)
      .format('wav')
      .save(wavPath)
      .on('end', () => resolve(wavPath))
      .on('error', reject);
  });
}

/**
 * Load WAV file as Float32Array
 * @param {string} wavPath - Path to WAV file
 * @returns {Promise<Float32Array>} - Audio data
 */
async function loadWavData(wavPath) {
  const buffer = await fs.readFile(wavPath);
  const decoded = await WavDecoder.decode(buffer);
  return decoded.channelData[0];
}

/**
 * Generic transcription function using transformers.js
 * @param {string} audioFilePath - Path to the audio file (MP3 supported)
 * @param {string} model - Model name (e.g., 'openai/whisper-base' or 'Oriserve/Whisper-Hindi2Hinglish-Swift')
 * @param {number} chunkLength - Chunk length in seconds for segmentation
 * @returns {Promise<Array>} - Array of transcription segments with timestamps
 */
async function transcribeWithJS(audioFilePath, model, chunkLength = 30) {
  if (!fs.existsSync(audioFilePath)) {
    throw new Error('Audio file not found');
  }

  console.log(`🚀 Starting JS Whisper transcription with model: ${model}`);

  let wavPath;
  try {
    // Convert to WAV if not already (assumes input is MP3 or WAV)
    wavPath = audioFilePath.endsWith('.wav') ? audioFilePath : await convertToWav(audioFilePath);
    const audioData = await loadWavData(wavPath);

    // Load pipeline (downloads model on first run, caches afterward)
    const transcriber = await pipeline('automatic-speech-recognition', model);

    // Transcribe with auto language detection and timestamps
    const result = await transcriber(audioData, {
      chunk_length_s: chunkLength,
      stride_length_s: chunkLength / 6,
      return_timestamps: true,  // Segment-level timestamps
      temperature: 0.0,
      do_sample: false,
      task: 'transcribe'
    });

    // Map chunks to segments
    const segments = (result.chunks || []).map(chunk => ({
      start: chunk.timestamp[0],
      end: chunk.timestamp[1],
      text: chunk.text.trim()
    })).filter(segment => segment.text.length > 0);

    console.log(`🚀 Transcription completed: ${segments.length} segments`);
    return segments;

  } catch (error) {
    console.error('👺 Transcription error:', error);
    throw new Error(`JS Whisper transcription failed: ${error.message}`);
  } finally {
    // Clean up temp WAV if created
    if (wavPath && wavPath !== audioFilePath) {
      await fs.remove(wavPath).catch(() => {});
    }
  }
}

/**
 * Process audio file using local Whisper model (JS version)
 * @param {string} audioFilePath - Path to the audio file
 * @param {string} model - Whisper model to use (base, small, medium, large)
 * @returns {Promise<Array>} - Array of transcription segments with timestamps
 */
async function processAudioWithWhisper(audioFilePath, model = 'base') {
  const fullModel = `openai/whisper-${model}`;
  return transcribeWithJS(audioFilePath, fullModel, 30);  // Larger chunks for standard models
}

/**
 * Alternative implementation using whisper with simpler output (JS version)
 * Uses multilingual model for better support of mixed languages
 */
async function processAudioWithWhisperSimple(audioFilePath, model = 'large') {
  const fullModel = `openai/whisper-${model}-v3`;  // Use v3 for better multilingual support
  return transcribeWithJS(audioFilePath, fullModel, 30);
}

/**
 * Process audio using specialized Hinglish Whisper model (JS version)
 * @param {string} audioFilePath - Path to the audio file
 * @param {string} model - Hugging Face model identifier for Hinglish
 * @returns {Promise<Array>} - Array of transcription segments
 */
async function processAudioWithHinglishWhisper(audioFilePath, model = 'Oriserve/Whisper-Hindi2Hinglish-Swift') {
  return transcribeWithJS(audioFilePath, model, 5);  // Smaller chunks like original (5s)
}

module.exports = {
  processAudioWithWhisper,
  processAudioWithHinglishWhisper,
  processAudioWithWhisperSimple
};
