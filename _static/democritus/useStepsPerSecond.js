export function useStepsPerSecond(target, callback) {
  let lastTime = performance.now()
  const targetDt = 1 / target * 1000
  let cancelled = false

  function loop() {
    if (cancelled) return

    const now = performance.now()
    const dt = (now - lastTime) / 1000
    lastTime = now

    const sps = 1 / dt
    callback({ sps, dt })

    const nextDelay = Math.max(0, targetDt - (performance.now() - now))
    setTimeout(loop, nextDelay)
  }

  setTimeout(loop, 0)

  return () => {
    cancelled = true
  }
}