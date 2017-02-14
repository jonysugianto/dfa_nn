package mlnn

import (
	"github.com/jonysugianto/mathlib/matrix"
)

type NNLayerConf struct {
	Size       int
	Inputsize  int
	Outputsize int
	Act_func   func([]float64) []float64
}

type NNLayer struct {
	Neurons     []*Neuron
	Activations *matrix.Vector
	Act_Func    func([]float64) []float64
}

func CreateNNLayer(size int, inputsize int, outputsize int) *NNLayer {
	var ret = new(NNLayer)
	for i := 0; i < size; i++ {
		var n = CreateNeuron(inputsize, outputsize)
		ret.Neurons = append(ret.Neurons, n)
	}
	ret.Activations = matrix.CreateVector(size)
	return ret
}

func (this *NNLayer) Init() {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Init()
	}
}

func (this *NNLayer) Activate(input *matrix.Vector) {
	var size = len(this.Neurons)
	var neurons_inputintegration []float64
	for i := 0; i < size; i++ {
		neurons_inputintegration = append(neurons_inputintegration, this.Neurons[i].InputIntegration(input))
	}
	this.Activations.Copy(this.Act_Func(neurons_inputintegration))
}

func (this *NNLayer) ComputeDFAError(target_error *matrix.Vector) {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].ComputeDFAError(target_error)
	}
}

func (this *NNLayer) SetError(target_error *matrix.Vector) {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].E = target_error.Values[i]
	}
}

func (this *NNLayer) Update(input *matrix.Vector, learningrate float64) {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Update(input, learningrate)
	}
}

func (this *NNLayer) WinnerNeuron() int {
	var maxindex int = 0
	var maxvalue = this.Activations.Values[0]
	var size = len(this.Activations.Values)
	for i := 1; i < size; i++ {
		if maxvalue < this.Activations.Values[i] {
			maxvalue = this.Activations.Values[i]
			maxindex = i
		}
	}
	return maxindex
}
