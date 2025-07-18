import { Injectable } from "@angular/core";
import { BehaviorSubject, Observable, throwError } from "rxjs";
import { tap, catchError } from "rxjs/operators";
import { SingleDateFilterModel } from "../models/singleFilter.model";
import {
  SinglePredResultService,
  SingleDatePredictionResponse
} from "./single-pred-result.service";

@Injectable({
  providedIn: 'root'
})

export class SinglePredictionDataFilPredService {
  private predictionResultSubject =
    new BehaviorSubject<SingleDatePredictionResponse | null>(null);
  public predictionResult$: Observable<SingleDatePredictionResponse | null> =
    this.predictionResultSubject.asObservable();

  private isLoadingSubject = new BehaviorSubject<boolean>(false);
  public isLoading$: Observable<boolean> = this.isLoadingSubject.asObservable();

  private errorSubject = new BehaviorSubject<any | null>(null);
  public error$: Observable<any | null> = this.errorSubject.asObservable();

  constructor(private singlePredResultService: SinglePredResultService) {}

  fetchAndStoreSingleDatePrediction(filters: SingleDateFilterModel): void {
    this.isLoadingSubject.next(true);
    this.errorSubject.next(null);
    this.predictionResultSubject.next(null);

    this.singlePredResultService
      .getSingleDatePrediction(filters)
      .pipe(
        tap((response: SingleDatePredictionResponse) => {
          this.predictionResultSubject.next(response);
          this.isLoadingSubject.next(false);
          console.log("Single date prediction response:", response);
        }),
        catchError((error) => {
          console.error("Error fetching single date prediction:", error);
          this.errorSubject.next(error);
          this.predictionResultSubject.next(null);
          this.isLoadingSubject.next(false);
          return throwError(() => error);
        })
      )
      .subscribe();
  }

  clearResults(): void {
    this.predictionResultSubject.next(null);
    this.errorSubject.next(null);
    this.isLoadingSubject.next(false);
  }

}
