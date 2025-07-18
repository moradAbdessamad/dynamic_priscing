import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { animate, state, style, transition, trigger } from '@angular/animations';
import { unitsOptions, targetPredictOptions, modelsOptions } from '../../models/filter.model';
import { SingleDateFilterModel, defaultSingleDateFilters } from '../../models/singleFilter.model';
import { SinglePredictionDataFilPredService } from '../../services/single-prediction-data-fil-pred.service';


@Component({
  selector: 'app-single-filters',
  standalone: true,
  imports: [CommonModule, FormsModule, ReactiveFormsModule],
  templateUrl: './single-filters.component.html',
  styleUrl: './single-filters.component.scss',
  animations: [
    trigger('expandCollapse', [
      state('expanded', style({
        height: '*',
        opacity: 1,
        visibility: 'visible'
      })),
      state('collapsed', style({
        height: '0',
        opacity: 0,
        visibility: 'hidden'
      })),
      transition('expanded <=> collapsed', [
        animate('250ms ease-in-out')
      ])
    ])
  ]
})
export class SingleFiltersComponent implements OnInit {
  filterForm!: FormGroup;
  filtersVisible = true;
  unitsOptions = unitsOptions;
  modelsOptions = modelsOptions;
  targetPredictOptions = targetPredictOptions;

  constructor(
    private fb: FormBuilder, 
    private SinglePredictionDataFilPredService: SinglePredictionDataFilPredService
  ) {}

  ngOnInit(): void {
    this.initForm();
  }

  initForm(): void {
    this.filterForm = this.fb.group({
      singleDate: [defaultSingleDateFilters.singleDate],
      unit: [defaultSingleDateFilters.unit],
      models: [defaultSingleDateFilters.models],
      targetPredict: [defaultSingleDateFilters.targetPredict],
      targetValueMax: [defaultSingleDateFilters.targetValueMax],
      targetValueMin: [defaultSingleDateFilters.targetValueMin]
    });
  }

  toggleFilters(): void {
    this.filtersVisible = !this.filtersVisible;
  }

  generateReport(): void {
    if (this.filterForm.valid) {
      const singleDateFilters = this.filterForm.value as SingleDateFilterModel;
      console.log('Generating report with single date filters:', singleDateFilters);
      
      this.SinglePredictionDataFilPredService.fetchAndStoreSingleDatePrediction(singleDateFilters);
    }
  }

  resetFilters(): void {
    this.filterForm.reset(defaultSingleDateFilters);
    this.SinglePredictionDataFilPredService.clearResults();
  }
}