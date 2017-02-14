package mlnn

import (
	"github.com/jonysugianto/mathlib/matrix"
)

type Perceptron struct {
	OutputLayer *NNLayer
	Loss_func   func(*matrix.Vector, *matrix.Vector) *matrix.Vector
}

func CreateMLP_DFA1(outputsize int, inputsize int) *Perceptron {
	var ret = new(Perceptron)
	ret.OutputLayer = CreateNNLayer(outputsize, inputsize, outputsize)
	return ret
}

func (this *Perceptron) Init() {
	this.OutputLayer.Init()
}

func (this *Perceptron) Activate(input *matrix.Vector) {
	this.OutputLayer.Activate(input)
}

func (this *Perceptron) Update(input *matrix.Vector, learningrate float64) {
	this.OutputLayer.Update(input, learningrate)
}

func (this *Perceptron) Learn(input *matrix.Vector, target *matrix.Vector, learningrate float64) *matrix.Vector {
	this.Activate(input)
	var target_error = this.Loss_func(target, this.OutputLayer.Activations)
	this.OutputLayer.SetError(target_error)
	this.Update(input, learningrate)
	return target_error
}
