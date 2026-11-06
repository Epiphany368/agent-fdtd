import {test,expect} from '@playwright/test';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';

test('polygon editing, ordered rotations, ellipse controls and GPU solve',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');
 await page.locator('[data-example="3d"]').click();await expect(page.getByLabel('dimension',{exact:true})).toHaveValue('3d');
 await page.getByLabel('time steps',{exact:true}).fill('80');await page.getByLabel('time steps',{exact:true}).press('Tab');
 await page.getByLabel('resource',{exact:true}).selectOption(process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu');
 await page.locator('[data-add="polygon"]').click();await page.getByRole('button',{name:'Edit polygon vertices',exact:true}).click();
 const d=page.locator('.geometry-dialog'),v=d.getByLabel('Polygon vertices',{exact:true});
 await v.fill('0,0\n1,1\n0,1\n1,0');await d.getByRole('button',{name:'Apply vertices',exact:true}).click();
 await expect(d.locator('[role="alert"]')).not.toBeEmpty();
 const before=await page.evaluate(()=>JSON.parse(localStorage.getItem('torchfdtd.project.v1')).structures.at(-1).vertices);
 expect(before).toHaveLength(3);
 await v.fill('-.6,-.4\n.6,-.4\n.6,0\n0,0\n0,.4\n-.6,.4');await d.getByRole('button',{name:'Apply vertices',exact:true}).click();await expect(d).not.toBeVisible();
 await page.getByLabel('first axis',{exact:true}).selectOption('x');
 await page.getByLabel('rotation 1',{exact:true}).fill('35');await page.getByLabel('rotation 1',{exact:true}).press('Tab');
 await page.getByLabel('second axis',{exact:true}).selectOption('y');
 await page.getByLabel('rotation 2',{exact:true}).fill('25');await page.getByLabel('rotation 2',{exact:true}).press('Tab');
 await page.getByLabel('x',{exact:true}).fill('0.65');await page.getByLabel('x',{exact:true}).press('Tab');
 await page.locator('[data-add="ring"]').click();await page.getByLabel('Elliptical radii',{exact:true}).check();
 await page.getByLabel('radius 2',{exact:true}).fill('0.65');await page.getByLabel('radius 2',{exact:true}).press('Tab');
 await page.getByLabel('theta start',{exact:true}).fill('30');await page.getByLabel('theta start',{exact:true}).press('Tab');
 await page.getByLabel('theta stop',{exact:true}).fill('280');await page.getByLabel('theta stop',{exact:true}).press('Tab');
 await page.getByLabel('x',{exact:true}).fill('-0.65');await page.getByLabel('x',{exact:true}).press('Tab');
 const p=await page.evaluate(()=>JSON.parse(localStorage.getItem('torchfdtd.project.v1')));
 expect(p.structures.at(-2).vertices).toHaveLength(6);expect(p.structures.at(-2).rotation_angles).toEqual([35,25,0]);
 const python=await page.request.post('/api/python',{data:p});expect(await python.text()).toContain('vertices');
 await page.screenshot({path:'results/ui-analytic-geometry.png',fullPage:true});
 await page.locator('#run-button').click();await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});expect(errors).toEqual([]);
});

test('synthetic ellipse FSP to editable native scene preserves input bytes',async({page})=>{
 const python=process.env.TORCHFDTD_TEST_PYTHON||(process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
 const raw=Buffer.from(execFileSync(python,['-c',"import sys,base64;sys.path.insert(0,'tests');from test_fsp_geometry import ellipse_fixture;print(base64.b64encode(ellipse_fixture()).decode())"],{encoding:'utf8'}).trim(),'base64');
 await page.goto('/');await expect(page.locator('#tree')).toContainText('waveguide');await page.locator('[data-action="fsp-native"]').click();
 await page.locator('#fsp-native-input').setInputFiles({name:'synthetic-ellipse.fsp',mimeType:'application/octet-stream',buffer:raw});
 await expect(page.locator('#fsp-native-status')).toContainText('Ready to open',{timeout:20000});
 const download=page.waitForEvent('download');await page.locator('[data-native="original"]').click();expect(fs.readFileSync(await (await download).path())).toEqual(raw);
 await page.locator('[data-native="load"]').click();
 await expect(page.locator('#tree')).toContainText('sphere');
 await expect.poll(()=>page.evaluate(()=>JSON.parse(localStorage.getItem('torchfdtd.project.v1'))?.structures?.[0]?.make_ellipsoid)).toBe(true);
 const p=await page.evaluate(()=>JSON.parse(localStorage.getItem('torchfdtd.project.v1')));expect(p.structures[0].make_ellipsoid).toBe(true);
 expect(p.structures[0].radius_2).toBeCloseTo(.21);expect(p.structures[0].radius_3).toBeCloseTo(.13);
 await page.getByLabel('resource',{exact:true}).selectOption(process.env.TORCHFDTD_TEST_CUDA?'cuda':'cpu');
 await page.locator('#run-button').click();await expect(page.locator('#mode-badge')).toHaveText('ANALYSIS',{timeout:90000});
});
