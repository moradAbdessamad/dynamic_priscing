import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { FilterModel } from '../models/filter.model'; 

export interface ApiPredictionRequest {
  fromDate: string;
  toDate: string;
  models: string;
  targetPredict: string;
  targetValueMin: number;
  targetValueMax: number;
  unit: string; 
}

export interface PredictionResult {
  [date: string]: {
    best_input_value: number;
    max_predicted_value: number;
  };
}

export interface PredictionResponse {
  model: string;
  result: PredictionResult; // Now directly an object, not a string
  status: string;
  target_type: string;
  unit: string;
}

export interface ParsedPredictionResponse {
  model: string;
  result: PredictionResult;
  status: string;
  target_type: string;
  unit: string;
}

@Injectable({
  providedIn: 'root'
})

export class PredResultService {
  private apiUrl = 'http://localhost:5300/api/calculate_predicted';

  constructor(private http: HttpClient) { }

  getPrediction(filterData: FilterModel): Observable<PredictionResponse> {
    const apiRequestData: ApiPredictionRequest = {
      ...filterData,
      unit: filterData.unit.toString() 
    };
    return this.http.post<PredictionResponse>(this.apiUrl, apiRequestData);
  }
}