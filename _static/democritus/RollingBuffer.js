// FIFO data structure
export class RollingBuffer {
  constructor(size) {
    this.size = size
    this.data = []
  }

  // clears the data
  reset() {
    this.data = []
  }

  // adds data to array and removes the first item if its too long
  scoot(next) {
    if (this.data.length > this.size) {
      this.data.shift()
    }
    this.data.push(next)
  }
}