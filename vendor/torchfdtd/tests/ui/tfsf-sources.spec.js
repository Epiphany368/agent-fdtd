import {test,expect} from '@playwright/test';

test('create a closed TFSF box, preview incidence and verify empty exterior on GPU',async({page})=>{
 const p=await (await page.request.get('/api/examples/scatterer')).json();
 p.structures=[];p.sources=[];p.region.size=[3.2,3.2,1];p.region.mesh=.1;p.region.pml_cells=4;p.region.steps=300;
 p.region.backend=process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu';p.region.cuda_kernel=process.env.TORCHFDTD_TEST_CUDA?'fused':'torch';
 p.region.material_sampling='yee';p.region.precision='float64';
 // The example's one-way sheet left periodic y faces; a TFSF box needs PML on every active face.
 p.region.boundaries.y_min.kind='pml';p.region.boundaries.y_max.kind='pml';
 p.monitors[0].center=[0,0,0];p.monitors[1].center=[1.1,1.1,0];
 await page.addInitScript(p=>localStorage.setItem('torchfdtd.project.v1',JSON.stringify(p)),p);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await page.locator('[data-add="tfsf"]').click();
 await page.getByLabel('x span',{exact:true}).fill('1.6');await page.getByLabel('x span',{exact:true}).press('Tab');
 await page.getByLabel('y span',{exact:true}).fill('1.6');await page.getByLabel('y span',{exact:true}).press('Tab');
 await page.getByLabel('propagation axis',{exact:true}).selectOption('y');
 await page.getByLabel('direction',{exact:true}).selectOption('-');
 await expect(page.getByLabel('x span',{exact:true})).toHaveValue('1.6');
 const pending=page.waitForResponse(r=>r.url().includes('/sources/')&&r.url().endsWith('/preview'));
 await page.getByRole('button',{name:'Preview time signal / spectrum',exact:true}).click();
 const preview=await (await pending).json();expect(preview.tfsf.normal).toBe('y');expect(preview.injections).toHaveLength(3);
 await page.locator('.source-dialog').getByRole('button',{name:'Close',exact:true}).click();
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();const id=(await (await submitted).json()).id;
 await expect(page.locator('#results-tree')).toContainText('field snapshots',{timeout:60000});
 const job=await (await page.request.get('/api/jobs/'+id)).json();
 const peaks=job.monitors.map(m=>Math.max(...m.signal.map(Math.abs)));
 expect(peaks[0]).toBeGreaterThan(.01);expect(peaks[1]/peaks[0]).toBeLessThan(1e-6);
 expect(job.summary.tfsf_boxes).toHaveLength(1);
 if(process.env.TORCHFDTD_TEST_CUDA)expect(job.summary.backend).toBe('cuda');
 await page.screenshot({path:'results/ui-tfsf-source.png',fullPage:true});expect(errors).toEqual([]);
});
