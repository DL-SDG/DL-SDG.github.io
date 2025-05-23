import { Vector2 } from "./Vector2.js"

// visualises the contents of a simulation
export class Visualiser {
  constructor(simulation, canvas) {
    this.simulation = simulation
    this.canvas = canvas
  
    // get canvas context
    this.ctx = this.canvas.getContext("2d")
  
    // get device pixel ratio
    const dpr = window.devicePixelRatio || 1
  
    // get CSS display width
    const rect = this.canvas.getBoundingClientRect()
    const displayWidth = rect.width
  
    // determine aspect ratio from simulation boundary
    const simWidth = this.simulation.boundary.x
    const simHeight = this.simulation.boundary.y
    const aspectRatio = simHeight / simWidth
  
    // compute display height based on aspect ratio
    const displayHeight = displayWidth * aspectRatio
  
    // set canvas size in actual pixels
    this.canvas.width = displayWidth * dpr
    this.canvas.height = displayHeight * dpr
  
    // set CSS size to match the aspect ratio
    this.canvas.style.width = `${displayWidth}px`
    this.canvas.style.height = `${displayHeight}px`
  
    // scale context for high DPI
    this.ctx.scale(dpr, dpr)
  
    // scale from simulation units to pixels
    this.scale = new Vector2(displayWidth / simWidth, displayHeight / simHeight)
  
    // pick the smaller scale if you want uniform scaling
    this.minimum_scale = Math.min(this.scale.x, this.scale.y)
  
    // for use when you want to draw something ~1px big
    this.single_pixel = 0.9 / this.scale
  
    // colors etc
    this.UNSELECTED_COLOR = "#0000ff"
    this.SELECTED_COLOR = "#ffff00"
  }

  onBoundaryUpdate(new_boundary) {
    // Get CSS display dimensions again
    const rect = this.canvas.getBoundingClientRect();
    const displayWidth = rect.width;
    const displayHeight = rect.height;
  
    // Recalculate scale from simulation units to pixels
    this.scale = new Vector2(
      displayWidth / new_boundary.x,
      displayHeight / new_boundary.y
    );
  
    // Update uniform scale
    this.minimum_scale = Math.min(this.scale.x, this.scale.y);
  
    // Update single pixel size in sim units (approximately 1px)
    this.single_pixel = 0.9 / this.scale;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
  }

  drawLine(start, end, width, color) {
    this.ctx.beginPath()
    this.ctx.moveTo(start.x * this.scale.x, start.y * this.scale.y)
    this.ctx.lineTo(end.x * this.scale.x, end.y * this.scale.y)
    this.ctx.strokeStyle = color
    this.ctx.lineWidth = width * this.minimum_scale
    this.ctx.stroke()
  }

  drawCircle(position, radius, color) {
    this.ctx.beginPath()
    this.ctx.arc(position.x * this.scale.x, position.y * this.scale.y, radius * this.minimum_scale, 0, Math.PI * 2)
    this.ctx.fillStyle = color
    this.ctx.fill()
    this.ctx.closePath()
  }

  update() {
    this.clear()

    // get a new boundary that is slightly large
    const slightlyBiggerBoundary = new Vector2(
      this.simulation.boundary.x + 1,
      this.simulation.boundary.y + 1
    )

    // draw horizontal grid lines
    for (let y = 0; y < slightlyBiggerBoundary.y + 1; y++) {
      this.drawLine(
        new Vector2(0, y),
        new Vector2(slightlyBiggerBoundary.x, y),
        this.single_pixel,
        "gray"
      )
    }

    // draw vertical grid lines
    for (let x = 0; x < slightlyBiggerBoundary.x + 1; x++) {
      this.drawLine(
        new Vector2(x, 0),
        new Vector2(x, slightlyBiggerBoundary.y),
        this.single_pixel,
        "gray"
      )
    }

    this.simulation.atoms.forEach(atom => {
      // represent the atom as a circle
      for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
          this.drawCircle(
            atom.position.add(this.simulation.boundary.multiply(new Vector2(x, y))),
            atom.radius,
            `${atom.selected ? this.SELECTED_COLOR : this.UNSELECTED_COLOR}${x == 0 && y == 0 ? "" : "99"}`
          )
        }
      }
    })
  }
}