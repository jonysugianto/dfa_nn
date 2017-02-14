package mlnn

import (
	"github.com/jonysugianto/mathlib/matrix"
)

//minimum one hidden layer
type MLNN struct {
	HiddenLayers []*NNLayer
	OutputLayer  *NNLayer
	Loss_func    func(*matrix.Vector, *matrix.Vector) *matrix.Vector

	Learningrate float64
}

func (this *MLNN) SizeHiddenLayer() int {
	return len(this.HiddenLayers)
}

func (this *MLNN) LastHiddenLayer() *NNLayer {
	return this.HiddenLayers[this.SizeHiddenLayer()-1]
}

func (this *MLNN) Init() {
	this.OutputLayer.Init()
	var size = this.SizeHiddenLayer()
	for i := 0; i < size; i++ {
		this.HiddenLayers[i].Init()
	}
}

func (this *MLNN) Outputs() *matrix.Vector {
	return this.OutputLayer.Activations
}

func (this *MLNN) Running(input *matrix.Vector) {
	this.HiddenLayers[0].Activate(input)
	var sizehiddenlayer = this.SizeHiddenLayer()
	if sizehiddenlayer > 1 {
		for i := 1; i < sizehiddenlayer; i++ {
			this.HiddenLayers[i].Activate(this.HiddenLayers[i-1].Activations)
		}
	}
	this.OutputLayer.Activate(this.LastHiddenLayer().Activations)
}

func (this *MLNN) ComputeTargetError(target *matrix.Vector) *matrix.Vector {
	return this.Loss_func(target, this.OutputLayer.Activations)
}

func (this *MLNN) ComputeDFAError(target_error *matrix.Vector) {
	this.OutputLayer.SetError(target_error)
	var sizehiddenlayer = this.SizeHiddenLayer()
	for i := 0; i < sizehiddenlayer; i++ {
		this.HiddenLayers[i].ComputeDFAError(target_error)
	}
}

func (this *MLNN) Update(input *matrix.Vector) {
	this.OutputLayer.Update(this.LastHiddenLayer().Activations, this.Learningrate)
	var sizehiddenlayer = this.SizeHiddenLayer()
	for i := 1; i < sizehiddenlayer; i++ {
		this.HiddenLayers[i].Update(this.HiddenLayers[i-1].Activations, this.Learningrate)
	}
	this.HiddenLayers[0].Update(input, this.Learningrate)
}

func (this *MLNN) Learning(input *matrix.Vector, target *matrix.Vector) *matrix.Vector {
	this.Running(input)
	var target_error = this.ComputeTargetError(target)
	this.ComputeDFAError(target_error)
	this.Update(input)
	return target_error
}

func (this *MLNN) LearningOld(input *matrix.Vector, target *matrix.Vector) *matrix.Vector {
	this.Running(input)
	var target_error = this.Loss_func(target, this.OutputLayer.Activations)
	this.OutputLayer.SetError(target_error)
	this.OutputLayer.Update(this.LastHiddenLayer().Activations, this.Learningrate)
	var sizehiddenlayer = this.SizeHiddenLayer()
	for i := 1; i < sizehiddenlayer; i++ {
		this.HiddenLayers[i].ComputeDFAError(target_error)
		this.HiddenLayers[i].Update(this.HiddenLayers[i-1].Activations, this.Learningrate)
	}
	this.HiddenLayers[0].ComputeDFAError(target_error)
	this.HiddenLayers[0].Update(input, this.Learningrate)
	return target_error
}
