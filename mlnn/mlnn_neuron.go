package mlnn

import (
	//	"fmt"
	"github.com/jonysugianto/mathlib/matrix"
	"github.com/jonysugianto/mathlib/random"
)

type Neuron struct {
	T float64
	W *matrix.Vector
	B *matrix.Vector
	E float64
}

func CreateNeuron(inputsize int, outputsize int) *Neuron {
	var ret = new(Neuron)
	ret.W = matrix.CreateVector(inputsize)
	ret.B = matrix.CreateVector(outputsize)
	return ret
}

func (this *Neuron) Init() {
	var rnvalues = random.RandomValues2(random.GaussianScaleMinusScale, this.B.Size(), 0.5)
	this.B.Copy(rnvalues)
	this.B.AddScalarI(0.1)
}

func (this *Neuron) InputIntegration(input *matrix.Vector) float64 {
	var suminput = this.W.Mul(input)
	var ret = suminput.Sum() + this.T
	return ret
}

func (this *Neuron) ComputeDFAError(target_error *matrix.Vector) {
	this.E = this.B.Mul(target_error).Sum()
}

func (this *Neuron) Update(input *matrix.Vector, learningrate float64) {
	var dT = -this.E
	this.T = this.T + learningrate*dT
	var dW = input.MulScalar(-this.E)
	//	fmt.Println("W size", this.W.Size(), "dw size", dW.Size())
	dW.MulScalarI(learningrate)
	this.W.AddI(dW)
}
