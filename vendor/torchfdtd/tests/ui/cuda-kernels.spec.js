import {test,expect} from '@playwright/test';

test('select fused CUDA in the UI and complete a native simulation',async({page})=>{
 const health=await (await page.request.get('/api/health')).json();
 test.skip(!health.cuda,'CUDA device unavailable');
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="3d"]').click();
 await expect(page.getByLabel('dimension',{exact:true})).toHaveValue('3d');
 await page.getByLabel('resource',{exact:true}).selectOption('cuda');
 await page.getByLabel('CUDA kernel',{exact:true}).selectOption('fused');
 await page.getByLabel('time steps',{exact:true}).fill('160');
 await page.getByLabel('time steps',{exact:true}).press('Tab');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const response=await submitted;
 expect(response.request().postDataJSON().region.backend).toBe('cuda');
 expect(response.request().postDataJSON().region.cuda_kernel).toBe('fused');
 const job=await response.json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:45000});
 await expect(page.locator('.run-summary')).toContainText('fused Yee / CPML');
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(result.summary.cuda_kernel).toBe('fused');
 expect(result.summary.field_peak).toBeGreaterThan(0);
 expect(result.summary.gpu).toBe(health.gpu);
 await page.screenshot({path:'results/ui-cuda-fused.png',fullPage:true});
 expect(errors).toEqual([]);
});
