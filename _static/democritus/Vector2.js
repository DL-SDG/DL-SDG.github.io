// Simple 2 dimensional vector
export class Vector2 {
  constructor(x, y) {
    this.x = x
    this.y = y
  }

  clone() {
    return new Vector2(this.x, this.y)
  }

  add(other) {
    return new Vector2(this.x + other.x, this.y + other.y)
  }

  subtract(other) {
    return new Vector2(this.x - other.x, this.y - other.y)
  }

  multiply(other) {
    return new Vector2(this.x * other.x, this.y * other.y)
  }

  divide(other) {
    return new Vector2(this.x / other.x, this.y / other.y)
  }

  dot(other) {
    return this.x * other.x + this.y * other.y
  }

  scale(value) {
    return new Vector2(this.x * value, this.y * value)
  }

  getArea() {
    return this.x * this.y
  }

  getMagnitude() {
    return Math.sqrt(Math.pow(this.x, 2) + Math.pow(this.y, 2))
  }

  // applies PBC to the vector, assuming the top left corner of the boundary is 0, 0
  applyPBC(boundary) {
    if (this.x < 0) {
      this.x = boundary.x - (Math.abs(this.x) % boundary.x)
    } else if (this.x > boundary.x) {
      this.x %= boundary.x
    }

    if (this.y < 0) {
      this.y = boundary.y - (Math.abs(this.y) % boundary.y)
    } else if (this.y > boundary.y) {
      this.y %= boundary.y
    }
  }

  // calculates the minimum distance to another vector under PBC
  getDistanceWithPBC(other, boundary) {
    let dx = this.x - other.x
    let dy = this.y - other.y

    // if distance can be made shorter by wrapping around the boundary, then do it
    if (dx > boundary.x / 2) dx -= boundary.x
    if (dx < -boundary.x / 2) dx += boundary.x
    if (dy > boundary.y / 2) dy -= boundary.y
    if (dy < -boundary.y / 2) dy += boundary.y

    return {
      delta: new Vector2(dx, dy),
      r: Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2))
    }
  }
}