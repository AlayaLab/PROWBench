// Normalize either the original multi-track document or a split camera index.
// Replay's in-memory interface remains {tracks, policies, ...metadata}.
export async function assembleCameras(document, readTrack) {
  if (document.schemaVersion !== 'generative-camera-index.v1') return document;
  const tracks = {}, policies = {};
  await Promise.all(Object.entries(document.tracks).map(async ([name, ref]) => {
    const data = await readTrack(ref.path, ref);
    if (data.schemaVersion !== 'generative-camera-track.v1' || data.track !== name ||
        !Array.isArray(data.frames) || data.frames.length !== ref.frameCount) {
      throw Error('Invalid camera track: ' + name);
    }
    tracks[name] = data.frames;
    if (document.hasPolicies) policies[name] = data.policy;
  }));
  // Keep recorded track order stable even when files load out of order.
  return {...document.metadata, tracks: Object.fromEntries(Object.keys(document.tracks).map(name => [name, tracks[name]])),
    ...(document.hasPolicies ? {policies} : {})};
}
