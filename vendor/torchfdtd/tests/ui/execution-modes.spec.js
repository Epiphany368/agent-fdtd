import {test,expect} from '@playwright/test';
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

// A second workbench server on a free loopback port, CPU only, with its own results directory under results/,
// which stop() removes once the server has exited.
async function serve(flags){
 const python=process.env.TORCHFDTD_TEST_PYTHON||(process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
 const port=await new Promise((resolve,reject)=>{const probe=net.createServer();probe.on('error',reject);probe.listen(0,'127.0.0.1',()=>{const {port}=probe.address();probe.close(()=>resolve(port));});});
 fs.mkdirSync('results',{recursive:true});
 const results=fs.mkdtempSync(path.resolve('results','admission-'));
 const child=spawn(python,['-m','torchfdtd.cli','serve','--port',String(port),...flags],{env:{...process.env,CUDA_VISIBLE_DEVICES:'-1',TORCHFDTD_RESULTS:results},stdio:'ignore'});
 const url=`http://127.0.0.1:${port}`;
 const exited=new Promise(resolve=>child.once('exit',resolve));
 const stop=async()=>{child.kill();await exited;fs.rmSync(results,{recursive:true,force:true});};
 for(let attempt=0;attempt<360;attempt++){
  try{const response=await fetch(url+'/api/health');if(response.ok)return {url,stop,health:await response.json()};}catch{}
  await new Promise(resolve=>setTimeout(resolve,250));
 }
 await stop();throw new Error('the second workbench server did not answer /api/health');
}

async function expectAdmission(page,url,health){
 expect(['fixed','memory']).toContain(health.admission);
 await page.goto(url+'/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="3d"]').click();
 const admission=page.locator('#admission'),properties=page.locator('#properties');
 await expect(page.locator('#execution-status')).toContainText('Auto →');
 if(health.admission==='memory'){ // torchfdtd serve --memory-admission
  expect(health.server_limits).toBeNull();
  await expect(admission).toHaveText('admission memory estimate');
  await expect(properties).not.toContainText('million cells');
 }else{
  expect(health.server_limits.resident_cells).toBe(8000000);
  await expect(admission).toHaveText('admission fixed server limits (8,000,000 resident cells)');
  await expect(properties).toContainText('75% of free GPU memory (80% of host memory on CPU) and at most 8 million cells;');
 }
}

test('GPU switch, memory modes, execution status line and a small Auto run',async({page})=>{
 const health=await (await page.request.get('/api/health')).json();
 expect(typeof health.cupy).toBe('boolean');
 expect(typeof health.host_available_bytes).toBe('number');
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="3d"]').click();
 await expect(page.getByLabel('dimension',{exact:true})).toHaveValue('3d');
 const gpu=page.getByLabel('GPU',{exact:true}),memory=page.getByLabel('memory',{exact:true}),status=page.locator('#execution-status');
 const resource=page.getByLabel('resource',{exact:true}),kernel=page.getByLabel('CUDA kernel',{exact:true}),monitorKernel=page.getByLabel('Frequency monitor kernel',{exact:true});
 await expect(memory).toHaveValue('auto');
 await expect(status).toContainText('Auto →');
 if(!health.cuda){
  await expect(gpu).toBeDisabled();
  await expect(gpu).not.toBeChecked();
  await expect(page.locator('.switch-row')).toHaveAttribute('title',/No CUDA device/);
  await expect(status).toContainText('Resident on CPU');
 }else{
  expect(typeof health.gpu_free_bytes).toBe('number');
  await expect(gpu).toBeEnabled();
  await expect(gpu).toBeChecked();
  await gpu.uncheck();
  await expect(resource).toHaveValue('cpu');
  await expect(kernel).toHaveValue('torch');
  await gpu.check();
  await expect(resource).toHaveValue('cuda');
  await expect(kernel).toHaveValue(health.cupy?'fused':'torch');
  await expect(monitorKernel).toHaveValue(health.cupy?'fused':'torch');
  await expect(status).toContainText('Auto → Resident on GPU');
  await expect(status).toContainText('free device memory');
  await gpu.uncheck();
  await expect(resource).toHaveValue('cpu');
 }
 await expect(status).toContainText('Auto → Resident on CPU');
 await expect(status).toContainText('80% of available host memory');
 await memory.selectOption('resident');
 await expect(status).toContainText('Resident on CPU');
 await expect(status).not.toContainText('Auto');
 await memory.selectOption('streamed_host');
 await expect(status).toContainText('Streamed through host memory on CPU');
 await expect(status).toContainText('host reservation');
 await expect(status).toContainText('slab 32 x depth 4');
 await memory.selectOption('streamed_disk');
 await expect(status).toContainText('Streamed through disk on CPU');
 await expect(status).toContainText('disk banks');
 await expect(status).toContainText('scratch');
 await memory.selectOption('auto');
 await expect(status).toContainText('Auto → Resident on CPU');
 await page.locator('[data-advanced-toggle]').click();
 await expect(page.locator('.advanced-execution')).toHaveClass(/open/);
 await expect(page.locator('.advanced-execution .advanced-body')).not.toHaveAttribute('inert');
 await resource.selectOption('cpu');
 await expect(gpu).not.toBeChecked();
 await page.getByLabel('time steps',{exact:true}).fill('120');
 await page.getByLabel('time steps',{exact:true}).press('Tab');
 await expect(status).toContainText('Auto → Resident on CPU');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const response=await submitted,body=response.request().postDataJSON();
 expect(body.region.execution_mode).toBe('auto');
 expect(body.region.backend).toBe('cpu');
 const job=await response.json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
 await expect(page.locator('.run-summary')).toContainText('Resident');
 await expect(page.locator('.run-summary')).toContainText('estimate');
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(result.execution.requested).toBe('auto');
 expect(result.execution.mode).toBe('resident');
 expect(result.execution.resident.fits).toBe(true);
 expect(result.summary.backend).toBe('cpu');
 expect(result.summary.steps).toBe(120);
 await page.screenshot({path:'results/ui-execution-modes.png',fullPage:true});
 expect(errors).toEqual([]);
});

test('the execution panel names the admission of each server mode',async({page,baseURL})=>{
 test.setTimeout(180000);
 // The server under test shows its own mode; a second server started here shows the other one.
 const health=await (await page.request.get('/api/health')).json();
 await expectAdmission(page,baseURL,health);
 const other=await serve(health.admission==='memory'?[]:['--memory-admission']);
 try{
  expect(other.health.admission).not.toBe(health.admission);
  await expectAdmission(page,other.url,other.health);
 }finally{await other.stop();}
});

test('streamed host run reports its policy, progress and one final snapshot',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="3d"]').click();
 await expect(page.getByLabel('dimension',{exact:true})).toHaveValue('3d');
 const gpu=page.getByLabel('GPU',{exact:true});
 if(await gpu.isEnabled())await gpu.uncheck();
 else{ // no CUDA device: the switch is disabled and the resource stays 'GPU if available'; pin the CPU explicitly
  const toggle=page.locator('[data-advanced-toggle]');
  if(await toggle.getAttribute('aria-expanded')!=='true')await toggle.click();
  await page.getByLabel('resource',{exact:true}).selectOption('cpu');
 }
 await expect(page.getByLabel('resource',{exact:true})).toHaveValue('cpu');
 await page.getByLabel('time steps',{exact:true}).fill('120');
 await page.getByLabel('time steps',{exact:true}).press('Tab');
 await page.getByLabel('memory',{exact:true}).selectOption('streamed_host');
 await expect(page.locator('#execution-status')).toContainText('Streamed through host memory on CPU');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const job=await (await submitted).json();
 const seen=new Set();
 await expect.poll(async()=>{seen.add(await page.locator('#status-text').innerText());return page.locator('#mode-badge').innerText();},{timeout:120000,intervals:[50]}).toBe('ANALYSIS');
 expect([...seen].some(text=>/Streaming block \d+ \/ 30 · tile \d+ \/ \d+/.test(text))).toBe(true);
 const summary=page.locator('.run-summary');
 await expect(summary).toContainText('Streamed through host memory');
 await expect(summary).toContainText('slab 32 × depth 4');
 await expect(summary).toContainText('DRAM banks');
 await expect(summary).toContainText('CPU tiles');
 await expect(summary).toContainText('host ');
 await expect(page.locator('#frame-label')).toHaveText('Step 120');
 await expect(page.locator('#messages')).toContainText('Execution: Streamed through host memory (slab 32 × depth 4, host banks)');
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(result.execution.mode).toBe('streamed_host');
 expect(result.execution.policy.checkpoints).toBe(0);
 expect(result.execution.policy.state_storage).toBe('host');
 expect(result.execution.report.blocks).toBe(30);
 expect(result.execution.report.completed_blocks).toBe(30);
 expect(result.summary.execution.snapshot).toBe('final step only');
 const fields=await (await page.request.get('/api/jobs/'+job.id+'/fields')).json();
 expect(fields.frame_steps).toEqual([120]);
 expect(result.monitors[0].signal.length).toBe(120);
 expect(errors).toEqual([]);
});

test('tiled mode shows its panel, the suggested overlap and the mismatch indicator of a run',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="scatterer"]').click();
 await expect(page.locator('#project-title')).toContainText('cylinder');
 for(const name of ['input','output']){await page.locator('[data-select]').filter({hasText:name}).click();await page.locator('[data-action="delete"]').first().click();}
 await expect(page.locator('#tree')).not.toContainText('output');
 await page.locator('[data-add="field"]').click();
 await page.locator('[data-path="center.0"]').fill('2');await page.locator('[data-path="center.0"]').press('Tab');
 await page.locator('[data-select]').filter({hasText:'source'}).click();
 await page.getByLabel('injection',{exact:true}).selectOption('soft');
 await page.locator('[data-path="size.1"]').fill('6');await page.locator('[data-path="size.1"]').press('Tab');
 await page.locator('[data-path="extend_through_pml"]').check();
 await page.locator('[data-action="region"]').first().click();
 await page.getByLabel('y min bc',{exact:true}).selectOption('pml');await page.getByLabel('y max bc',{exact:true}).selectOption('pml');
 const gpu=page.getByLabel('GPU',{exact:true});
 if(await gpu.isEnabled())await gpu.uncheck();
 else{ // no CUDA device: the switch is disabled and the resource stays 'GPU if available'; pin the CPU explicitly
  const toggle=page.locator('[data-advanced-toggle]');
  if(await toggle.getAttribute('aria-expanded')!=='true')await toggle.click();
  await page.getByLabel('resource',{exact:true}).selectOption('cpu');
 }
 await expect(page.getByLabel('resource',{exact:true})).toHaveValue('cpu');
 await page.getByLabel('time steps',{exact:true}).fill('400');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await expect(page.locator('.tiled-panel')).toHaveCount(0);
 await page.getByLabel('memory',{exact:true}).selectOption('tiled');
 const panel=page.locator('.tiled-panel'),status=page.locator('#execution-status');
 await expect(panel).toBeVisible();
 await expect(panel).toContainText('tiled adjoint is Python only');
 await expect(status).toContainText('Tiled (approximate) on CPU');
 await page.getByLabel('tile size',{exact:true}).fill('2');await page.getByLabel('tile size',{exact:true}).press('Tab');
 await page.getByLabel('tile overlap',{exact:true}).fill('0.5');await page.getByLabel('tile overlap',{exact:true}).press('Tab');
 await expect(status).toContainText('3 tiles (3) of 2 um with 0.5 um overlap');
 await expect(page.locator('#tiled-suggestion')).toContainText(/\d µm = \d.* spread \+ .* absorber/);
 await expect(status).toContainText('below the suggested');
 await page.getByLabel('Propagate to a focal plane',{exact:true}).check();
 await expect(page.getByLabel('propagation distance',{exact:true})).toHaveValue('10');
 await expect(status).toContainText('focal plane at 10 um');
 const submitted=page.waitForResponse(r=>r.url().endsWith('/api/jobs')&&r.request().method()==='POST');
 await page.locator('#run-button').click();
 const response=await submitted,body=response.request().postDataJSON();
 expect(body.region.execution_mode).toBe('tiled');
 expect(body.region.tiling).toEqual({size_um:2,overlap_um:0.5,max_angle_deg:45,propagation_um:10,allow_approximate:false});
 expect(body.sources[0].extend_through_pml).toBe(true);
 const job=await response.json();
 await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:120000});
 const summary=page.locator('.run-summary');
 await expect(summary).toContainText('Tiled (approximate) · 3 tiles (3) of 2.00 µm · overlap 0.500 µm · focal plane +10 µm');
 await expect(summary.locator('.indicator')).toContainText(/max mismatch \d\.\d{3}/);
 await expect(summary.locator('.mismatch-table tbody tr')).toHaveCount(2);
 await expect(summary.locator('.mismatch-note')).toContainText('error indicator of an approximate method');
 await expect(page.locator('#results-tree')).toContainText('Stitched plane');
 await expect(page.locator('#field-label')).toContainText('focal plane |E|^2 / peak at 10 um');
 const result=await (await page.request.get('/api/jobs/'+job.id)).json();
 expect(result.execution.mode).toBe('tiled');
 expect(result.execution.report.pairs.length).toBe(2);
 expect(typeof result.execution.indicator.max_mismatch_center).toBe('number');
 expect(result.flux_monitors.map(m=>m.id.endsWith('-focal'))).toEqual([false,true]);
 const fields=await (await page.request.get('/api/jobs/'+job.id+'/fields')).json();
 expect(fields.frames.length).toBe(2);
 await page.screenshot({path:'results/ui-execution-modes-tiled.png',fullPage:true});
 expect(errors).toEqual([]);
});
