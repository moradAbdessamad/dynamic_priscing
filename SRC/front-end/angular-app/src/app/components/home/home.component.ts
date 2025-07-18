import { Component } from '@angular/core';
import { HeaderComponent } from '../header/header.component';
import { FiltersComponent } from '../filters/filters.component';
import { PredResultComponent } from '../pred-result/pred-result.component';
import { DynamicTableComponent } from '../dynamic-table/dynamic-table.component';
import { SideBarComponent } from '../side-bar/side-bar.component';
import { RouterOutlet } from '@angular/router';

const DUMMY_TABLE_DATA = {
  "2025-06-01": { "best_input": 8.5, "max_output": 820000, "min_error": 2.1 },
  "2025-06-02": { "best_input": 11.0, "max_output": 830500, "min_error": 1.8 },
  "2025-06-03": { "best_input": 9.7, "max_output": 840200, "min_error": 2.4 },
  "2025-06-04": { "best_input": 14.3, "max_output": 850800, "min_error": 1.5 },
  "2025-06-05": { "best_input": 12.8, "max_output": 860100, "min_error": 1.9 },
  "2025-06-06": { "best_input": 10.5, "max_output": 870450, "min_error": 2.2 },
  "2025-06-07": { "best_input": 13.0, "max_output": 880000, "min_error": 1.7 },
  "2025-06-08": { "best_input": 13.0, "max_output": 880000, "min_error": 1.7 }
};

@Component({
  selector: 'app-home',
  imports: [HeaderComponent, SideBarComponent, RouterOutlet],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})

export class HomeComponent {
  sampleDataForTable = DUMMY_TABLE_DATA;
}