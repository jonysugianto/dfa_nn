package cnn

import (
	//	"fmt"
	"github.com/jonysugianto/mathlib/conv"
	"github.com/jonysugianto/mathlib/matrix"
	"github.com/jonysugianto/mathlib/random"
)

type ConvConf struct {
	Act_func         func(float64) float64
	ConvInputSize    int
	Input_w          int
	Input_h          int
	Conv_w           int
	Conv_h           int
	Stride_w         int
	Stride_h         int
	Pool_h           int
	Pool_w           int
	TargetOutputSize int
	NumberNeurons    int
}

type ConvNeuron struct {
	T           float64
	W           *matrix.Vector
	B           *matrix.MatrixVector
	Input       *matrix.MatrixVector
	Activations *matrix.MatrixFloat
	Err         *matrix.MatrixFloat
	Dw          *matrix.MatrixVector
	Dt          *matrix.MatrixFloat
}

func CreateConvNeuron(conf ConvConf) *ConvNeuron {
	var ret = new(ConvNeuron)
	ret.W = matrix.CreateVector(conf.Conv_w * conf.Conv_h * conf.ConvInputSize)
	var conv_outputsize_w = conv.ComputeConvOutputSize(conf.Input_w, conf.Conv_w, conf.Stride_w)
	var conv_outputsize_h = conv.ComputeConvOutputSize(conf.Input_h, conf.Conv_h, conf.Stride_h)
	ret.B = matrix.CreateMatrixVector(conv_outputsize_w, conv_outputsize_h, conf.TargetOutputSize)
	ret.Input = matrix.CreateMatrixVector(conf.Input_w, conf.Input_h, conf.ConvInputSize)
	ret.Activations = matrix.CreateMatrixFloat(conv_outputsize_w, conv_outputsize_h)
	ret.Err = matrix.CreateMatrixFloat(conv_outputsize_w, conv_outputsize_h)
	ret.Dw = matrix.CreateMatrixVector(conv_outputsize_w, conv_outputsize_h, conf.ConvInputSize)
	ret.Dt = matrix.CreateMatrixFloat(conv_outputsize_w, conv_outputsize_h)
	return ret
}

func (this *ConvNeuron) Init() {
	var rowsize = this.B.Row()
	var colsize = this.B.Col()
	for i := 0; i < rowsize; i++ {
		for j := 0; j < colsize; j++ {
			var rnvalues = random.RandomValues2(random.GaussianScaleMinusScale, this.B.Get(i, j).Size(), 0.5)
			this.B.Get(i, j).Copy(rnvalues)
			this.B.Get(i, j).AddScalarI(0.1)
		}
	}
}

func (this *ConvNeuron) Activate(input *matrix.MatrixVector, act_func func(float64) float64) {
	var rowsize = input.Row()
	var colsize = input.Col()
	for i := 0; i < rowsize; i++ {
		for j := 0; j < colsize; j++ {
			var conv_input = input.Get(i, j)
			var suminput = this.W.Mul(conv_input).Sum() + this.T
			this.Activations.Put(i, j, act_func(suminput))
		}
	}
}

func (this *ConvNeuron) ComputeDFAError(target_error *matrix.Vector) {
	var rowsize = this.B.Row()
	var colsize = this.B.Col()
	//	fmt.Println("B", this.B.Get(0, 0).Size(), " target_error", target_error.Size())
	for i := 0; i < rowsize; i++ {
		for j := 0; j < colsize; j++ {
			var e = this.B.Get(i, j).Mul(target_error).Sum()
			this.Err.Put(i, j, e)
		}
	}
}

func (this *ConvNeuron) Compute_Dw_Dt(input *matrix.MatrixVector) {
	var rowsize = input.Row()
	var colsize = input.Col()
	for i := 0; i < rowsize; i++ {
		for j := 0; j < colsize; j++ {
			var e = this.Err.Get(i, j)
			this.Dt.Put(i, j, -e)
			var dW = input.Get(i, j).MulScalar(-e)
			this.Dw.Put(i, j, dW)
		}
	}
}

func (this *ConvNeuron) Update_W_T(learningrate float64) {
	//	fmt.Println("conv neuron update w t")
	var rowsize = this.Dw.Row()
	var colsize = this.Dw.Col()
	var dt float64
	var dw = matrix.CreateVector(this.W.Size())
	for i := 0; i < rowsize; i++ {
		for j := 0; j < colsize; j++ {
			dw.AddI(this.Dw.Get(i, j))
			dt = dt + this.Dt.Get(i, j)
		}
	}
	//	fmt.Println("dt", dt, "dw", dw.Values)
	dw.MulScalarI(learningrate)
	this.W.AddI(dw)
	this.T = this.T + learningrate*dt
}
