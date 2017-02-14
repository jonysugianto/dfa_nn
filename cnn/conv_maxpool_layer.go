package cnn

import (
	"github.com/jonysugianto/mathlib/matrix"
)

type ConvMaxpoolLayer struct {
	ConvolutionLayer *ConvLayer
	MaxPoolingLayer  *MaxpoolLayer
}

func CreateConvMaxpoolLayer(conf ConvConf) *ConvMaxpoolLayer {
	var ret = new(ConvMaxpoolLayer)
	ret.ConvolutionLayer = CreateConvLayer(conf)
	ret.MaxPoolingLayer = CreateMaxPoolLayer(conf)
	return ret
}

func (this *ConvMaxpoolLayer) Init() {
	this.ConvolutionLayer.Init()
}

func (this *ConvMaxpoolLayer) Outputs() []*matrix.MatrixFloat {
	return this.MaxPoolingLayer.Activations
}

func (this *ConvMaxpoolLayer) ActivateFromMatrixFloat(raw_input *matrix.MatrixFloat) {
	this.ConvolutionLayer.ActivateFromMatrixFloat(raw_input)
	this.MaxPoolingLayer.MaxPooling(this.ConvolutionLayer.GetActivations())
}

func (this *ConvMaxpoolLayer) ActivateFromListOfMatrixFloat(raw_inputs []*matrix.MatrixFloat) {
	this.ConvolutionLayer.ActivateFromListOfMatrixFloat(raw_inputs)
	this.MaxPoolingLayer.MaxPooling(this.ConvolutionLayer.GetActivations())
}

func (this *ConvMaxpoolLayer) ComputeDFAError(target_error *matrix.Vector) {
	this.ConvolutionLayer.ComputeDFAError(target_error)
}

func (this *ConvMaxpoolLayer) Compute_Dw_Dt() {
	this.ConvolutionLayer.Compute_Dw_Dt()
}

func (this *ConvMaxpoolLayer) Update_W_T(learningrate float64) {
	this.ConvolutionLayer.Update_W_T(learningrate)
}
