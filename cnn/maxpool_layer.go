package cnn

import (
	//	"fmt"
	"math"

	"github.com/jonysugianto/mathlib/conv"
	"github.com/jonysugianto/mathlib/matrix"
)

type MaxpoolLayer struct {
	Activations []*matrix.MatrixFloat
	Pool_w      int
	Pool_h      int
}

func CreateMaxPoolLayer(conf ConvConf) *MaxpoolLayer {
	var ret = new(MaxpoolLayer)
	var conv_outputsize_w = conv.ComputeConvOutputSize(conf.Input_w, conf.Conv_w, conf.Stride_w)
	var conv_outputsize_h = conv.ComputeConvOutputSize(conf.Input_h, conf.Conv_h, conf.Stride_h)
	ret.Pool_w = conf.Pool_w
	ret.Pool_h = conf.Pool_h
	for i := 0; i < conf.NumberNeurons; i++ {
		ret.Activations = append(ret.Activations, matrix.CreateMatrixFloat(conv_outputsize_w, conv_outputsize_h))
	}
	return ret
}

func (this *MaxpoolLayer) pooling(input *matrix.MatrixFloat) *matrix.MatrixFloat {
	var newrow = int((float64(input.Row()) / float64(this.Pool_w)) + 0.499999)
	var newcol = int((float64(input.Col()) / float64(this.Pool_h)) + 0.499999)
	var ret = matrix.CreateMatrixFloat(newrow, newcol)
	var maxRow = input.Row() - 1
	var maxCol = input.Col() - 1

	for r := 0; r < newrow; r++ {
		for c := 0; c < newcol; c++ {
			var maxvalue = 0.0
			for pw := 0; pw < this.Pool_w; pw++ {
				for ph := 0; ph < this.Pool_h; ph++ {
					var row_pos = int(math.Min(float64(maxRow), float64(r*this.Pool_w+pw)))
					var col_pos = int(math.Min(float64(maxCol), float64(c*this.Pool_h+ph)))
					if maxvalue < input.Get(row_pos, col_pos) {
						maxvalue = input.Get(row_pos, col_pos)
					}
				}
			}
			//			fmt.Println("maxvalue", maxvalue)
			ret.Put(r, c, maxvalue)
		}
	}
	return ret
}

func (this *MaxpoolLayer) MaxPooling(inputs []*matrix.MatrixFloat) {
	var size = len(inputs)
	for i := 0; i < size; i++ {
		this.Activations[i] = this.pooling(inputs[i])
	}
}
