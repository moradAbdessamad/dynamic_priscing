import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { DefaultDashboardComponent } from './components/default-dashboard/default-dashboard.component';
import { DefaultPageComponent } from './components/default-page/default-page.component';
import { SingleDashboardComponent } from './components/single-dashboard/single-dashboard.component';

export const routes: Routes = [
  {
    path: '',
    component: HomeComponent,
    children: [
      {
        path: '',
        component: DefaultDashboardComponent,
      },
      {
        path: 'single-date-pred',
        component: SingleDashboardComponent
      },
      {
        path: '**',
        component: DefaultPageComponent,
      }
    ]
  }
];