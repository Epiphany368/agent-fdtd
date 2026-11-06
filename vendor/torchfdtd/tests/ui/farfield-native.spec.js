import {test,expect} from '@playwright/test';
import {mkdir,readFile,writeFile} from 'node:fs/promises';

test('native six-face CPU run exports the same closed-box radiation shown in the browser',async({page})=>{
 test.setTimeout(180000);
 const project=JSON.parse(await readFile(new URL('../../examples/native_farfield.json',import.meta.url),'utf8'));
 await mkdir('results',{recursive:true});
 const retained=process.env.TORCHFDTD_FARFIELD_JOB;
 await page.addInitScript(({project,retained})=>{localStorage.setItem('torchfdtd.project.v1',JSON.stringify(project));if(retained)localStorage.setItem('torchfdtd.activeJob',retained);else localStorage.removeItem('torchfdtd.activeJob');},{project,retained});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto('/');
 let id=retained;
 if(!id){
  const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
  await page.locator('#run-button').click();const response=await submitted;expect(response.ok()).toBe(true);id=(await response.json()).id;
  await writeFile('results/native-farfield-job.json',JSON.stringify({id},null,2)+'\n');
 }
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 const job=await (await page.request.get(`/api/jobs/${id}`)).json();expect(job.status).toBe('completed');
 const npz=await page.request.get(`/api/jobs/${id}/download`);expect(npz.ok()).toBe(true);await writeFile('results/native-farfield-result.npz',await npz.body());
 const before=await (await page.request.get('/api/jobs')).json();
 await page.locator('[data-action="flux-results"]').first().click();await page.getByRole('button',{name:'Closed-box far field',exact:true}).click();
 const d=page.locator('.farfield-dialog');await expect(d).toBeVisible();
 await expect(d.getByLabel('X minimum face')).toHaveValue('x_min');await expect(d.getByLabel('Z upper bound (µm)')).toHaveValue('0.6');
 await d.getByRole('heading',{name:'Closed-box far field',exact:true}).scrollIntoViewIfNeeded();await page.screenshot({path:'results/ui-farfield-setup.png',fullPage:true});
 await d.getByLabel('Confirm homogeneous closed surface').check();await d.getByRole('button',{name:'Calculate far field'}).click();
 await expect(d.getByRole('heading',{name:'Angular radiation pattern'})).toBeVisible({timeout:30000});
 await expect(d.getByRole('status')).toContainText('Total fields');
 const download=page.waitForEvent('download');await d.getByRole('button',{name:'Export result JSON'}).click();
 const content=await readFile(await (await download).path(),'utf8'),result=JSON.parse(content);await writeFile('results/native-farfield-result.json',content+'\n');
 expect(result.directions).toHaveLength(19*36);expect(result.report.points).toBe(6*12*12);expect(result.zero_pattern).toBe(false);
 expect(result.field_kind).toBe('total');expect(result.amplitude_units).toBe('reduced E * s * m');expect(result.intensity_units).toBe('reduced E*H * s^2 * m^2 / sr');
 expect(content).not.toMatch(/NaN|Infinity/);expect(Math.max(...result.relative_intensity.flat())).toBeCloseTo(1,12);
 // Coarse native-monitor integration check only, not a mesh-convergence claim.
 let squareError=0,squareReference=0;
 result.theta_deg.forEach((theta,i)=>{const exact=Math.sin(theta*Math.PI/180)**2;result.phi_deg.forEach((_,j)=>{squareError+=(result.relative_intensity[i][j]-exact)**2;squareReference+=exact**2;});});
 const patternRelativeL2=Math.sqrt(squareError/squareReference);expect(patternRelativeL2).toBeLessThan(.05);
 const csvDownload=page.waitForEvent('download');await d.getByRole('button',{name:'Export direction CSV'}).click();
 const csv=await readFile(await (await csvDownload).path(),'utf8');expect(csv.trim().split('\n')).toHaveLength(19*36+1);await writeFile('results/native-farfield-directions.csv',csv);
 await d.getByRole('heading',{name:'Angular radiation pattern'}).scrollIntoViewIfNeeded();await page.screenshot({path:'results/ui-farfield-result.png',fullPage:true});
 const after=await (await page.request.get('/api/jobs')).json();expect(after.map(j=>j.id)).toEqual(before.map(j=>j.id));expect(errors).toEqual([]);
 await writeFile('results/native-farfield-workflow.json',JSON.stringify({id,summary:job.summary,pattern_relative_l2:patternRelativeL2,predeclared_pattern_limit:.05,points:result.report.points,directions:result.directions.length,postprocessing_started_fdtd:false},null,2)+'\n');
});
