import { Vector2 } from './Vector2.js'

// Atom class that stores the position, velocity and force of an atom
export class Atom {
  constructor(x, y, radius) {
    this.position = new Vector2(x, y)
    this.velocity = new Vector2(0, 0)
    this.force = new Vector2(0, 0)

    this.radius = radius
    this.selected = false
  }
}