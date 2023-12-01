// This is test for smaller range of value

import Alpaca from "@alpacahq/alpaca-trade-api";
import brain from "brain.js";
import "dotenv/config";

const alpaca = new Alpaca({
  keyId: process.env.ALPACA_KEY,
  secretKey: process.env.ALPACA_SECRET,
  paper: true,
});
const symbol = "INDF";

async function main() {
  const data = await fetchHistoricalData();
  const prediction = await makePrediction(data);
  console.log("Scaled prediction:", prediction);
}
main();

async function fetchHistoricalData() {
  try {
    const data = alpaca.getBarsV2(symbol, {
      start: "2023-01-01",
      end: "2023-09-15",
      limit: 1000,
      timeframe: "1D",
    });

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
  let min = Infinity;
  let max = -Infinity;
  for (let d of data) {
    min = Math.min(min, d.close);
    max = Math.max(max, d.close);
  }

  const trainingData = data.map(scaleDown);

  const net = new brain.recurrent.LSTMTimeStep();
  net.train([trainingData.slice(0, data.length - 1)], {
    learningRate: 0.005,
    errorThresh: 0.00001,
  });

  const slice = trainingData.slice(0, data.length - 6);
  const prediction = net.forecast(slice, 5);

  return prediction.map(scaleUp);

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
