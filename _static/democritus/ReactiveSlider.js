export class ReactiveSlider {
  constructor(name, initial_value, formatter, onInput) {
    this.input_element = document.getElementById(`${name}_slider`)
    this.output_element = document.getElementById(`${name}_value`)

    this.formatter = formatter
    this.onInput = onInput

    this.input_element.value = initial_value
    this.updateOutput()

    this.input_element.addEventListener("input", () => {
      this.updateOutput()
      this.onInput(this.getValue())
    })
  }

  getValue() {
    return parseInt(this.input_element.value)
  }

  updateOutput() {
    this.output_element.innerText = this.formatter(this.getValue())
  }
}