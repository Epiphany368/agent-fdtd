import {test,expect} from '@playwright/test';

test('select shared CUDA frequency monitors and collect six-component spectra',async({page})=>{
 const health=await (await page.request.get('/api/health')).json();
 test.skip(!health.cuda,'CUDA device unavailable');
 const p=await (await page.request.get('/api/examples/scatterer')).json();
 p.region.backend='cuda';p.region.cuda_kernel='fused';p.region.steps=160;p.region.material_sampling='yee';
 p.monitors.push({id:'spectral-plane',name:'Spectral plane',kind:'field',normal:'x',center:[.7,0,0],size:[0,1,1],
  spectrum:{sampling:'frequency',frequency_points:5,apodization:'none'}});
 await page.addInitScript(p=>localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p)),p);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await page.getByLabel('Frequency monitor kernel',{exact:true}).selectOption('fused');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const response=await submitted;
 expect(response.request().postDataJSON().region.cuda_monitor_kernel).toBe('fused');
 const job=await response.json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:45000});
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(result.summary.cuda_monitor_kernel).toBe('fused');
 expect(result.summary.field_peak).toBeGreaterThan(0);
 expect(result.summary.gpu).toBe(health.gpu);
 const flux=result.flux_monitors;
 expect(flux).toHaveLength(1);expect(flux[0].frequency_thz).toHaveLength(5);
 expect(flux[0].flux.every(Number.isFinite)).toBeTruthy();
 expect(Math.max(...flux[0].flux.map(Math.abs))).toBeGreaterThan(0);
 await page.screenshot({path:'results/ui-cuda-monitors.png',fullPage:true});
 expect(errors).toEqual([]);
});
