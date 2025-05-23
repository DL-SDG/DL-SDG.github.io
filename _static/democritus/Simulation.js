import { BOLTZMAN_CONSTANT } from './Constants.js'
import { Vector2 } from "./Vector2.js"

// simulation control
export class Simulation {
  constructor(atoms, boundary) {
    this.atoms = atoms
    this.boundary = boundary

    // calculate the cutoff radius
    this.cutoff_radius = 0.5 * Math.min(this.boundary.x, this.boundary.y)

    // calculate alpha and beta for shifted lennard jones potential
    this.alpha = this.getAlpha()
    this.beta = this.getBeta()

    // ensure that all atoms are within the boundary
    this.atoms.forEach(atom => atom.position.applyPBC(this.boundary))

    // no steps have taken place yet
    this.step = 0

    // how much time each step represents
    this.full_step = 0.001
    this.half_step = this.full_step / 2

    // to cap the speed of the atoms
    this.maximum_gamma = 1000000

    // desired values
    this.desired_temperature = 500

    // actual values
    this.temperature = NaN
    this.potential_energy = NaN
    this.total_energy = NaN
    this.virial = NaN
    this.pressure = NaN
    this.density = NaN
  }

  getAlpha() {
    return 24 * ((2 * Math.pow(this.cutoff_radius, -12)) - Math.pow(this.cutoff_radius, -6)) / this.cutoff_radius
  }

  getBeta() {
    return 4 * (Math.pow(this.cutoff_radius, -12) - Math.pow(this.cutoff_radius, -6)) + this.alpha * this.cutoff_radius
  }

  setBoundary(new_boundary) {
    const scale = new_boundary.divide(this.boundary)

    this.alpha = this.getAlpha()
    this.beta = this.getBeta()

    this.atoms.forEach(atom => {
      atom.position = atom.position.multiply(scale)
    })

    this.boundary = new_boundary
  }

  updateForces() {
    this.atoms.forEach(atom => { atom.force = new Vector2(0, 0)})

    // reset statistics
    this.potential_energy = 0
    this.virial = 0

    for (let i = 0; i < this.atoms.length; i++) {
      const first = this.atoms.at(i)
      for (let j = i + 1; j < this.atoms.length; j++) {
        const second = this.atoms.at(j)

        const { delta, r } = first.position.getDistanceWithPBC(second.position, this.boundary)
        
        if (r < this.cutoff_radius) {
          // calculate gamma
          let gamma = 24 * Math.pow(r, -2) * ((2 * Math.pow(r, -12)) - Math.pow(r, -6)) - this.alpha * Math.pow(r, -1)
        
          // limit gamma
          gamma = Math.min(gamma, this.maximum_gamma)
          
          //apply forces
          first.force = first.force.add(delta.scale(gamma))
          second.force = second.force.subtract(delta.scale(gamma))

          // calculate statistics
          this.potential_energy += 4 * (Math.pow(r, -12) - Math.pow(r, -6)) + this.alpha * r - this.beta
          this.virial -= gamma * Math.pow(r, 2)
        }
      }
    }
  }

  update() {
    // first halfstep
    this.atoms.forEach(atom => {
      atom.velocity = atom.velocity.add(atom.force.scale(this.half_step))
      atom.position = atom.position.add(atom.velocity.scale(this.full_step))
    })

    this.updateForces()

    // second halfstep, also calculate the kinetic energy
    const kinetic_energy = this.atoms.reduce((total, atom) => {
      atom.velocity = atom.velocity.add(atom.force.scale(this.half_step))
      atom.position.applyPBC(this.boundary)

      return total + Math.pow(atom.velocity.x, 2) + Math.pow(atom.velocity.y, 2)
    }, 0) / 2

    this.temperature = kinetic_energy / (BOLTZMAN_CONSTANT * this.atoms.length)
    this.total_energy = kinetic_energy + this.potential_energy
    this.density = this.atoms.length / this.boundary.getArea()
    this.pressure = (kinetic_energy - (this.virial / 2)) / this.density

    if (this.step % 10 == 0) {
      const temperature_scaling = Math.sqrt(this.desired_temperature / this.temperature)

      const average_velocity = this.atoms.reduce(
        (momentum, atom) => momentum.add(atom.velocity),
        new Vector2(0, 0)
      ).scale(1 / this.atoms.length)

      this.atoms.forEach(atom => {
        atom.velocity = atom.velocity.subtract(average_velocity).scale(temperature_scaling)
      })
    }

    this.step++
  }
}