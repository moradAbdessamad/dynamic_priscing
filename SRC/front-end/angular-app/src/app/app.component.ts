import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';


@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  template: `
    <div class="main-app-container">
      <router-outlet />
    </div>
  `,
  styles: [
    `
    .main-app-container {
      width: 100%;
      height: 100%;
      display: flex;
      justify-content: center;
    }
    `
  ],
})
export class AppComponent {
  
}
