package main

import (
	"fmt"
	"math"
	"strconv"

	"github.com/jonysugianto/dfa_nn/cnn"
	"github.com/jonysugianto/dfa_nn/minist_example/reader"
	"github.com/jonysugianto/dfa_nn/mlnn"
	"github.com/jonysugianto/mathlib/matrix"
	"github.com/jonysugianto/mathlib/neuralfunction"
	"github.com/jonysugianto/mathlib/random"
)

func WinnerIndex(outputs *matrix.Vector) int {
	var maxindex int = 0
	var maxvalue = outputs.Values[0]
	var size = len(outputs.Values)
	for i := 1; i < size; i++ {
		if maxvalue < outputs.Values[i] {
			maxvalue = outputs.Values[i]
			maxindex = i
		}
	}
	return maxindex
}

func main() {
	var lblfilename = "/data/neocortexid/golang/minist/train-labels.idx1-ubyte"
	var imagefilename = "/data/neocortexid/golang/minist/train-images.idx3-ubyte"
	var labels = reader.ReadMinistLabel(lblfilename)
	var dataset = reader.ReadMinistImage(imagefilename)
	var targets []*matrix.Vector

	var size = len(labels)
	for i := 0; i < size; i++ {
		targets = append(targets, reader.ConvLabelToTargetVector(labels[i]))
	}

	var builder = cnn.CreateBuilder()
	builder.MlnnBuilder.SetLearningrate(0.05).SetLoss_func(
		neuralfunction.AbsPower2).SetOutputLayer(
		mlnn.NNLayerConf{Size: 10, Inputsize: 100, Outputsize: 10, Act_func: neuralfunction.SigmoidArray}).AddHiddenLayer(
		mlnn.NNLayerConf{Size: 100, Inputsize: 490, Outputsize: 10, Act_func: neuralfunction.SigmoidArray})

	builder.MlCnnBuilder.SetLearningrate(0.005).AddCnnMaxpoolLayer(
		cnn.ConvConf{Act_func: neuralfunction.Sigmoid, ConvInputSize: 1, Input_w: 28, Input_h: 28,
			Conv_w: 3, Conv_h: 3, Stride_w: 1, Stride_h: 1,
			Pool_h: 2, Pool_w: 2, TargetOutputSize: 10, NumberNeurons: 5}).AddCnnMaxpoolLayer(
		cnn.ConvConf{Act_func: neuralfunction.Sigmoid, ConvInputSize: 5, Input_w: 13, Input_h: 13,
			Conv_w: 3, Conv_h: 3, Stride_w: 1, Stride_h: 1,
			Pool_h: 2, Pool_w: 2, TargetOutputSize: 10, NumberNeurons: 10})

	var network, _ = builder.Build()
	network.Init()

	var epoch = 2 * size
	var error_per_periode float64
	var correct_per_periode float64
	for i := 0; i < epoch; i++ {
		var rn = random.Random(size)
		var target_error = network.Learning(dataset[rn].AsMatrixFloat(28, 28), targets[rn])
		var outputs = network.Outputs()
		var winner_nr = WinnerIndex(outputs)
		if winner_nr == labels[rn] {
			correct_per_periode = correct_per_periode + 1
		}
		//fmt.Println("target error", target_error)
		target_error.SquareI()
		error_per_periode = error_per_periode + math.Sqrt(target_error.Sum())
		if (i % 1000) == 0 {
			fmt.Print(strconv.Itoa(i))
			fmt.Print(" Error per 1000 ")
			fmt.Printf("%.2f", error_per_periode)
			fmt.Print(" Correct per 1000 ")
			fmt.Printf("%.2f", correct_per_periode)
			fmt.Println()
			fmt.Println(outputs)
			fmt.Println(targets[rn].Values)
			error_per_periode = 0
			correct_per_periode = 0
		}
	}
}
