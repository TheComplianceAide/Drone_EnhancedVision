/**
 * Default Node Media Server configuration for FLV streaming.
 * Serves streams published to rtmp://127.0.0.1/live/<stream_key>
 * and makes them available over HTTP at:
 *   http://127.0.0.1:8000/live/<stream_key>.flv
 */
module.exports = {
  rtmp: {
    port: 1935,
    chunk_size: 60000,
    gop_cache: true,
    ping: 30,
    ping_timeout: 60
  },
  http: {
    port: 8000,
    allow_origin: '*'
  },
  trans: {
    ffmpeg: 'ffmpeg',
    tasks: [
      {
        app: 'live',
        mp4: false,
        hls: false,
        dash: false
      }
    ]
  }
};