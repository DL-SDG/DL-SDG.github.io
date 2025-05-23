export class Countdown {
  constructor(limit, callback) {
    this.limit = limit
    this.progress = 0

    this.callback = callback
  }

  update(dt, context) {
    this.progress += dt

    if (this.progress > this.limit) {
      this.progress -= this.limit

      this.callback(context)
    }
  }
}