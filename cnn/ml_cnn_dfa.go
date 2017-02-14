package cnn

import (
	"github.com/jonysugianto/mathlib/matrix"
)

type MultiLayerCnnDfa struct {
	Layers       []*ConvMaxpoolLayer
	Learningrate float64
}

func (this *MultiLayerCnnDfa) Init() {
	for i := 0; i < len(this.Layers); i++ {
		this.Layers[i].Init()
	}
}

func (this *MultiLayerCnnDfa) LastCnnLayer() *ConvMaxpoolLayer {
	return this.Layers[len(this.Layers)-1]
}

func (this *MultiLayerCnnDfa) Activate(input *matrix.MatrixFloat) {
	this.Layers[0].ActivateFromMatrixFloat(input)
	var size = len(this.Layers)
	for i := 1; i < size; i++ {
		this.Layers[i].ActivateFromListOfMatrixFloat(this.Layers[i-1].Outputs())
	}
}

func (this *MultiLayerCnnDfa) ComputeDFAError(target_error *matrix.Vector) {
	var size = len(this.Layers)
	for i := 0; i < size; i++ {
		this.Layers[i].ComputeDFAError(target_error)
	}
}

func (this *MultiLayerCnnDfa) Compute_Dw_Dt() {
	var size = len(this.Layers)
	for i := 0; i < size; i++ {
		this.Layers[i].Compute_Dw_Dt()
	}
}

func (this *MultiLayerCnnDfa) Update_W_T() {
	var size = len(this.Layers)
	for i := 0; i < size; i++ {
		this.Layers[i].Update_W_T(this.Learningrate)
	}
}

func (this *MultiLayerCnnDfa) Outputs() *matrix.Vector {
	var lastoutput = this.LastCnnLayer().MaxPoolingLayer.Activations
	return matrix.ListOfMatrixFloatAsVector(lastoutput)
}
