package mlnn

import (
	//	"fmt"
	"github.com/jonysugianto/mathlib/matrix"
)

type MLNNBuilder struct {
	OutputLayerConf  NNLayerConf
	HiddenLayerConfs []NNLayerConf

	Loss_func    func(*matrix.Vector, *matrix.Vector) *matrix.Vector
	Learningrate float64
}

func (this *MLNNBuilder) SetOutputLayer(OutputLayerConf NNLayerConf) *MLNNBuilder {
	this.OutputLayerConf = OutputLayerConf
	return this
}

func (this *MLNNBuilder) AddHiddenLayer(HiddenLayer NNLayerConf) *MLNNBuilder {
	this.HiddenLayerConfs = append(this.HiddenLayerConfs, HiddenLayer)
	return this
}

func (this *MLNNBuilder) SetLearningrate(lr float64) *MLNNBuilder {
	this.Learningrate = lr
	return this
}

func (this *MLNNBuilder) SetLoss_func(lf func(*matrix.Vector, *matrix.Vector) *matrix.Vector) *MLNNBuilder {
	this.Loss_func = lf
	return this
}

func (this *MLNNBuilder) Build() (*MLNN, error) {
	var ret = new(MLNN)
	ret.Learningrate = this.Learningrate
	ret.Loss_func = this.Loss_func
	ret.OutputLayer = CreateNNLayer(this.OutputLayerConf.Size, this.OutputLayerConf.Inputsize, this.OutputLayerConf.Outputsize)
	ret.OutputLayer.Act_Func = this.OutputLayerConf.Act_func
	var size = len(this.HiddenLayerConfs)
	for i := 0; i < size; i++ {
		var hl = CreateNNLayer(this.HiddenLayerConfs[i].Size, this.HiddenLayerConfs[i].Inputsize, this.HiddenLayerConfs[i].Outputsize)
		hl.Act_Func = this.HiddenLayerConfs[i].Act_func
		//	fmt.Println("layer", i, len(hl.Neurons), hl.Neurons[0].W.Size())
		ret.HiddenLayers = append(ret.HiddenLayers, hl)
	}
	return ret, nil
}
