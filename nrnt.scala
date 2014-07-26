import scala.math.{exp, pow, random, sqrt, tanh}
/**
 * Created by Daniel Piet on 5/18/2014.
 */
object NrNt {
  def main(args: Array[String]) = {

    var patterns = new Array[Array[Array[Double]]](4)
    patterns = Array(
      Array(Array(1,0), Array(0)),
      Array(Array(0,1), Array(0)),
      Array(Array(1,1), Array(1)),
      Array(Array(0,0), Array(0))
    )

    val nn = new NeuralNetwork(2,2,1, HyperbolicTangent)
    // iterations, learning rate, momentum factor
    nn.train(patterns, 1000, 0.5 ,0.1)

    for (p <- 0 until patterns.size) {
      println(f"${patterns(p)(0).deep.mkString(", ")} -> ${"%1.5f".format(nn.update(patterns(p)(0))(0))}")
    }
  }
}

abstract class ActFn {
  def calc(arg: Double) : Double
  def dcalc(arg: Double) : Double
}

object Logistic extends ActFn {
  // sigmoid function
  override def calc(arg: Double) : Double = {
    1d / (1d + exp(-arg))
  }
  // derivative in terms of y
  override def dcalc(y: Double) : Double = {
    y - pow(y,2)
  }
}

object HyperbolicTangent extends ActFn {
  override def calc(arg: Double) : Double = {
    tanh(arg)
  }
  override def dcalc(y: Double) : Double = {
    1 - pow(y,2)
  }
}

class NeuralNetwork (var inputNumber: Int, val hiddenNumber: Int, val outputNumber: Int, val activation: ActFn) {
  inputNumber = inputNumber + 1 // bias
  var activationInput = Array.fill(inputNumber){ 1.0 }
  var activationHidden = Array.fill(hiddenNumber){ 1.0 }
  var activationOutput = Array.fill(outputNumber){ 1.0 }
  var weightsInput = makeMatrix(inputNumber, hiddenNumber, .2)
  var weightsOutput = makeMatrix(hiddenNumber, outputNumber, 2.0)
  var inputChange = makeMatrix(inputNumber, hiddenNumber, 0.0)
  var outputChange = makeMatrix(hiddenNumber, outputNumber, 0.0)

  // make a matrix of size I x J filled with random values between -randVal and +randVal
  def makeMatrix(I: Int, J: Int, randVal: Double): Array[Array[Double]] = {
    Array.fill(I,J){ random * (randVal * 2) - randVal }
  }

  def update(inputs: Array[Double]): Array[Double] = {
    if (inputs.size != inputNumber - 1) {
      throw new Exception("Wrong input size")
    }

    // input activations
    Array.copy(inputs, 0, activationInput, 0, inputNumber - 1)

    // hidden activations
    for (j <- 0 until hiddenNumber) {
      var sum = 0.0
      for (i <- 0 until inputNumber) {
        sum  = sum + activationInput(i) * weightsInput(i)(j)
      }
      activationHidden(j) = activation.calc(sum)
    }

    // output activations
    for (k <- 0 until outputNumber) {
      var sum = 0.0
      for (l <- 0 until hiddenNumber) {
        sum = sum + activationHidden(l) * weightsOutput(l)(k)
      }
      activationOutput(k) = activation.calc(sum)
    }

    activationOutput
  }

  def backPropagate(targets: Array[Double], N: Double, M: Double) = {
    if (targets.size != outputNumber) {
      throw new Exception("Wrong input size")
    }

    // output error
    val outputDelta = Array.fill(outputNumber){ 0.0 }
    for (k <- 0 until outputNumber) {
      val error = targets(k) - activationOutput(k)
      outputDelta(k) = activation.dcalc(activationOutput(k)) * error
    }

    // hidden error
    val hiddenDelta = Array.fill(hiddenNumber){ 0.0 }
    for (j <- 0 until hiddenNumber) {
      var error = 0.0
      for (k <- 0 until outputNumber) {
        error = error + outputDelta(k) * weightsOutput(j)(k)
      }
      hiddenDelta(j) = activation.dcalc(activationHidden(j)) * error
    }

    // output weights
    for (j <- 0 until hiddenNumber) {
      for (k <- 0 until outputNumber) {
        val change = outputDelta(k) * activationHidden(j)
        weightsOutput(j)(k) = weightsOutput(j)(k) + N*change + M*outputChange(j)(k)
        outputChange(j)(k) = change
      }
    }

    // input weights
    for (i <- 0 until inputNumber) {
      for (j <- 0 until hiddenNumber) {
        val change = hiddenDelta(j) * activationInput(i)
        weightsInput(i)(j) = weightsInput(i)(j) + N*change + M*inputChange(i)(j)
        inputChange(i)(j) = change
      }
    }

    // error calc
    var outputError = 0.0
    for (k <- 0 until targets.size) {
      outputError = outputError + 0.5 * pow((targets(k) - activationOutput(k)), 2)
    }
    outputError
  }

  def train(patterns: Array[Array[Array[Double]]], iterations: Int, N: Double, M: Double) = {
    for (i <- 0 until iterations) {
      var error = 0.0
      for (p <- 0 until patterns.size) {
        val inputs = patterns(p)(0)
        val targets = patterns(p)(1)
        update(inputs)
        error = error + backPropagate(targets,N,M)
      }
      if (i % 100 == 0) {
        println(s" error %${"%1.5f".format(error)}")
      }
    }
  }
}

