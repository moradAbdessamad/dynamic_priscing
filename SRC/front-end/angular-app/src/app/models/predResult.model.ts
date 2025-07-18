export interface ApiPredictionRequest {
  fromDate: string;
  toDate: string;
  models: string;
  targetPredict: string;
  targetValueMin: number;
  targetValueMax: number;
  unit: string; // API expects unit as string
}

export interface PredictionResult {
  [date: string]: {
    best_input: number;
    max_output: number;
  };
}

export interface PredictionResponse {
  result: string; // This is a JSON string
  status: string;
}

export interface ParsedPredictionResponse {
  result: PredictionResult;
  status: string;
}