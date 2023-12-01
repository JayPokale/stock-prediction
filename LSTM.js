import Alpaca from "@alpacahq/alpaca-trade-api";
import brain from "brain.js";
import "dotenv/config";

// SIgn to ALPACA Trade API
const alpaca = new Alpaca({
  keyId: process.env.ALPACA_KEY,
  secretKey: process.env.ALPACA_SECRET,
  paper: true,
});
const symbol = "INDF";

async function main() {
  // Collect Hestoric Data
  const data = await fetchHistoricalData();

  // Create Prediction by LSTM Model
  const prediction = await makePrediction(data);

  // Display Prediction
  console.log("Scaled prediction:", prediction);
}
main();

async function fetchHistoricalData() {
  try {
    // Get data of given range
    const data = alpaca.getBarsV2(symbol, {
      start: "2013-09-15",
      end: "2023-09-15",
      limit: 1000,
      timeframe: "1D",
    });

    // Filter out data
    const got = [];
    for await (const b of data) {
      got.push({
        open: b.OpenPrice,
        high: b.HighPrice,
        low: b.LowPrice,
        close: b.ClosePrice,
        VWAP: b.VWAP,
      });
    }

    return got;
  } catch (error) {
    console.error("Error fetching historical data:", error);
    throw error;
  }
}

async function makePrediction(data) {
  // Get minimum and maximum closing value for given set of data
  let min = Infinity;
  let max = -Infinity;
  for (let d of data) {
    min = Math.min(min, d.close);
    max = Math.max(max, d.close);
  }

  // Scale all vlue in range 0 to 1
  const trainingData = data.map(scaleDown);

  // Run LSTM on time step data
  const net = new brain.recurrent.LSTMTimeStep();
  net.train([trainingData.slice(0, data.length - 1)], {
    learningRate: 0.005,
    errorThresh: 0.00001,
  });

  const slice = trainingData.slice(0, data.length - 6);
  const prediction = net.forecast(slice, 5);

  // Return scaled up data
  return prediction.map(scaleUp);

  // Utility functions to Convert data between:
  // 1. Original range to [0,1]
  // 2. [0,1] to Original range
  function scaleDown(x) {
    return {
      open: scaleDownFunc(x.open),
      high: scaleDownFunc(x.high),
      low: scaleDownFunc(x.low),
      close: scaleDownFunc(x.close),
      VWAP: scaleDownFunc(x.VWAP),
    };
  }

  function scaleUp(x) {
    return {
      open: scaleUpFunc(x.open),
      high: scaleUpFunc(x.high),
      low: scaleUpFunc(x.low),
      close: scaleUpFunc(x.close),
      VWAP: scaleUpFunc(x.VWAP),
    };
  }

  function scaleDownFunc(x) {
    x -= (max + min) / 2;
    x *= 0.8 / (max - min);
    x += 0.5;
    return x;
  }

  function scaleUpFunc(x) {
    x -= 0.5;
    x *= (max - min) / 0.8;
    x += (max + min) / 2;
    return x;
  }
}
