package cnn

import (
	"github.com/jonysugianto/mathlib/conv"
	"github.com/jonysugianto/mathlib/matrix"
)

type ConvLayer struct {
	Neurons    []*ConvNeuron
	ConvInputs *matrix.MatrixVector
	Act_func   func(float64) float64
	Input_w    int
	Input_h    int
	Conv_w     int
	Conv_h     int
	Stride_w   int
	Stride_h   int
}

func CreateConvLayer(conf ConvConf) *ConvLayer {
	var ret = new(ConvLayer)
	for i := 0; i < conf.NumberNeurons; i++ {
		ret.Neurons = append(ret.Neurons, CreateConvNeuron(conf))
	}
	ret.Act_func = conf.Act_func
	ret.Input_w = conf.Input_w
	ret.Input_h = conf.Input_h
	ret.Conv_w = conf.Conv_w
	ret.Conv_h = conf.Conv_h
	ret.Stride_w = conf.Stride_w
	ret.Stride_h = conf.Stride_h
	return ret
}

func (this *ConvLayer) Init() {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Init()
	}
}

func (this *ConvLayer) ActivateFromMatrixFloat(raw_input *matrix.MatrixFloat) {
	this.ConvInputs = conv.ConvInputsFromMatrixFloat(raw_input,
		this.Input_w, this.Input_h, this.Conv_w, this.Conv_h, this.Stride_w, this.Stride_h)
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Activate(this.ConvInputs, this.Act_func)
	}
}

func (this *ConvLayer) ActivateFromListOfMatrixFloat(raw_inputs []*matrix.MatrixFloat) {
	this.ConvInputs = conv.ConvInputsFromListOfMatrixFloat(raw_inputs,
		this.Input_w, this.Input_h, this.Conv_w, this.Conv_h, this.Stride_w, this.Stride_h)
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Activate(this.ConvInputs, this.Act_func)
	}
}

func (this *ConvLayer) ComputeDFAError(target_error *matrix.Vector) {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].ComputeDFAError(target_error)
	}
}

func (this *ConvLayer) Compute_Dw_Dt() {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Compute_Dw_Dt(this.ConvInputs)
	}
}

func (this *ConvLayer) Update_W_T(learningrate float64) {
	var size = len(this.Neurons)
	for i := 0; i < size; i++ {
		this.Neurons[i].Update_W_T(learningrate)
	}
}

func (this *ConvLayer) GetActivations() []*matrix.MatrixFloat {
	var size = len(this.Neurons)
	var ret []*matrix.MatrixFloat
	for i := 0; i < size; i++ {
		this.Neurons[i].Activate(this.ConvInputs, this.Act_func)
		ret = append(ret, this.Neurons[i].Activations)
	}
	return ret
}
