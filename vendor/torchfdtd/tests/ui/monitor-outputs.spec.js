import {test,expect} from '@playwright/test';
import fs from 'node:fs';

test('select spectral outputs, precision, strides and local apodization, then run CUDA',async({page})=>{
 const health=await (await page.request.get('/api/health')).json();test.skip(!health.cuda,'CUDA unavailable');
 const p=await (await page.request.get('/api/examples/3d')).json();
 p.region.backend='cuda';p.region.cuda_kernel='fused';p.region.cuda_monitor_kernel='fused';p.region.steps=160;
 p.global_monitor={sampling:'chebyshev',chebyshev_nodes:'lobatto',frequency_points:5,apodization:'full'};
 p.monitors=[{id:'selected',name:'Selective plane',kind:'field',normal:'x',center:[.7,0,0],size:[0,1,1],
  use_global_monitor:true,inherit_apodization:true,spectrum:{sampling:'frequency',apodization:'none'}}];
 await page.addInitScript(p=>localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p)),p);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');
 await page.locator('#tree').getByText('Selective plane',{exact:true}).click();
 await page.getByLabel('Inherit global apodization',{exact:true}).uncheck();
 await expect(page.getByLabel('apodization',{exact:true})).toBeEnabled();
 await page.getByLabel('DFT accumulation precision',{exact:true}).selectOption('float64');
 await page.getByLabel('downsample y',{exact:true}).fill('2');await page.getByLabel('downsample y',{exact:true}).press('Tab');
 await page.getByLabel('DFT time downsample',{exact:true}).fill('2');await page.getByLabel('DFT time downsample',{exact:true}).press('Tab');
 for(const c of ['Ex','Ey','Hx','Hy','Px','Py','Pz'])await page.getByLabel('Record '+c,{exact:true}).uncheck();
 await page.getByLabel('Record signed flux',{exact:true}).uncheck();
 await page.getByLabel('spatial interpolation',{exact:true}).selectOption('nearest');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();const response=await submitted,job=await response.json();
 const sent=response.request().postDataJSON().monitors[0];
 expect(sent.record_fields).toEqual(['Ez','Hz']);expect(sent.record_poynting).toEqual([]);
 expect(sent.downsample_xyz).toEqual([1,2,1]);expect(sent.time_downsample).toBe(2);expect(sent.dft_precision).toBe('float64');
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:45000});
 const field=await page.request.get(`/api/jobs/${job.id}/field-monitors/selected?component=Ez`);expect(field.ok()).toBeTruthy();
 expect((await field.json()).magnitude.flat().some(v=>v>0)).toBeTruthy();
 const missing=await page.request.get(`/api/jobs/${job.id}/field-monitors/selected?component=Ex`);expect(missing.status()).toBe(422);
 expect((await (await page.request.get('/api/jobs/'+job.id)).json()).flux_monitors).toEqual([]);
 await page.screenshot({path:'results/ui-monitor-outputs.png',fullPage:true});expect(errors).toEqual([]);
});

test('synthetic FSP plane imports, preserves original bytes and runs on GPU',async({page})=>{
 test.skip(!process.env.TORCHFDTD_SPECTRAL_FSP,'Synthetic spectral FSP path required');
 const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');
 await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-action="fsp-native"]').click();
 await page.locator('#fsp-native-input').setInputFiles(process.env.TORCHFDTD_SPECTRAL_FSP);
 await expect(page.locator('#fsp-native-status')).toContainText('Ready to open',{timeout:20000});
 await expect(page.locator('.native-issues')).toContainText('Native DFT');
 const download=page.waitForEvent('download');await page.locator('[data-native="original"]').click();
 expect(fs.readFileSync(await (await download).path())).toEqual(fs.readFileSync(process.env.TORCHFDTD_SPECTRAL_FSP));
 await page.locator('[data-native="load"]').click();
 await page.locator('#tree').getByText('monitor',{exact:true}).click();
 await expect(page.getByLabel('DFT accumulation precision',{exact:true})).toHaveValue('float64');
 await expect(page.getByLabel('DFT time downsample',{exact:true})).toHaveValue('3');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();const job=await (await submitted).json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:45000});
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(result.summary.gpu).toContain('5880');expect(result.flux_monitors).toHaveLength(1);
 expect(result.flux_monitors[0].flux.some(v=>Math.abs(v)>0)).toBeTruthy();
 await page.screenshot({path:'results/ui-fsp-selective-monitor.png',fullPage:true});expect(errors).toEqual([]);
});
