/**
 * Stable Diffusion WebUI Automation - JavaScript/Node.js Version
 * 
 * This script provides programmatic access to the Stable Diffusion WebUI API
 * for automated image generation in Node.js/JavaScript environments.
 * 
 * Usage:
 *   node automate_sd.js --prompt "your prompt" --output image.png
 *   node automate_sd.js --prompt "landscape" --steps 30 --width 1024 --height 576
 * 
 * Or use as a module:
 *   const { StableDiffusionAPI } = require('./automate_sd.js');
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Configuration
const DEFAULT_URL = process.env.SD_WEBUI_URL || 'http://127.0.0.1:7860';
const API_BASE = '/sdapi/v1';

/**
 * Make HTTP request to WebUI API
 */
function apiRequest(endpoint, method = 'GET', data = null) {
    return new Promise((resolve, reject) => {
        const url = new URL(endpoint, DEFAULT_URL);
        const options = {
            hostname: url.hostname,
            port: url.port || 7860,
            path: url.pathname,
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(body));
                } catch (e) {
                    resolve(body);
                }
            });
        });

        req.on('error', reject);
        
        if (data) {
            req.write(JSON.stringify(data));
        }
        req.end();
    });
}

/**
 * Stable Diffusion WebUI API Wrapper
 */
class StableDiffusionAPI {
    constructor(baseUrl = DEFAULT_URL) {
        this.baseUrl = baseUrl;
    }

    /**
     * Wait for WebUI API to be ready
     */
    async waitForApi(timeout = 60) {
        console.log(`Waiting for WebUI at ${this.baseUrl}...`);
        const start = Date.now();
        
        while (Date.now() - start < timeout * 1000) {
            try {
                await this.getOptions();
                console.log('WebUI API is ready!');
                return true;
            } catch (e) {
                await new Promise(r => setTimeout(r, 2000));
            }
        }
        console.log('Timeout waiting for WebUI');
        return false;
    }

    /**
     * Get available options
     */
    async getOptions() {
        return apiRequest(`${API_BASE}/options`, 'GET');
    }

    /**
     * Get available models
     */
    async getModels() {
        return apiRequest(`${API_BASE}/sd-models`, 'GET');
    }

    /**
     * Get available samplers
     */
    async getSamplers() {
        return apiRequest(`${API_BASE}/samplers`, 'GET');
    }

    /**
     * Generate images from text prompt (txt2img)
     */
    async txt2img(options = {}) {
        const {
            prompt = '',
            negative_prompt = '',
            steps = 20,
            cfg_scale = 7.0,
            width = 512,
            height = 512,
            seed = -1,
            batch_size = 1,
            sampler_name = 'Euler a'
        } = options;

        console.log(`Generating: '${prompt}' (steps=${steps}, cfg=${cfg_scale})`);
        
        const payload = {
            prompt,
            negative_prompt,
            steps,
            cfg_scale,
            width,
            height,
            seed,
            batch_size,
            sampler_name
        };

        const result = await apiRequest(`${API_BASE}/txt2img`, 'POST', payload);
        return result.images || [];
    }

    /**
     * Generate images from image + text prompt (img2img)
     */
    async img2img(options = {}) {
        const {
            prompt = '',
            negative_prompt = '',
            init_images = [],
            steps = 20,
            cfg_scale = 7.0,
            denoising_strength = 0.75,
            seed = -1,
            sampler_name = 'Euler a'
        } = options;

        console.log(`Img2Img: '${prompt}'`);
        
        const payload = {
            prompt,
            negative_prompt,
            init_images,
            steps,
            cfg_scale,
            denoising_strength,
            seed,
            sampler_name
        };

        const result = await apiRequest(`${API_BASE}/img2img`, 'POST', payload);
        return result.images || [];
    }

    /**
     * Read and encode image to base64
     */
    readImageAsBase64(imagePath) {
        const buffer = fs.readFileSync(imagePath);
        return buffer.toString('base64');
    }

    /**
     * Save base64 image to file
     */
    saveImage(base64Data, outputPath) {
        const buffer = Buffer.from(base64Data, 'base64');
        const dir = path.dirname(outputPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(outputPath, buffer);
        return outputPath;
    }

    /**
     * Generate and save image in one call
     */
    async generate(options = {}) {
        const { output = 'output.png', ...txt2imgOptions } = options;
        const images = await this.txt2img(txt2imgOptions);
        if (images.length > 0) {
            this.saveImage(images[0], output);
            console.log(`Saved: ${output}`);
            return output;
        }
        throw new Error('No images generated');
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const options = {
        prompt: '',
        negative: '',
        output: 'output.png',
        steps: 20,
        cfg: 7.0,
        width: 512,
        height: 512,
        seed: -1,
        sampler: 'Euler a',
        url: DEFAULT_URL,
        img2img: null,
        denoise: 0.75
    };

    // Parse arguments
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        switch (arg) {
            case '--prompt':
            case '-p':
                options.prompt = args[++i];
                break;
            case '--negative':
            case '-n':
                options.negative = args[++i];
                break;
            case '--output':
            case '-o':
                options.output = args[++i];
                break;
            case '--steps':
            case '-s':
                options.steps = parseInt(args[++i]);
                break;
            case '--cfg':
                options.cfg = parseFloat(args[++i]);
                break;
            case '--width':
                options.width = parseInt(args[++i]);
                break;
            case '--height':
                options.height = parseInt(args[++i]);
                break;
            case '--seed':
                options.seed = parseInt(args[++i]);
                break;
            case '--sampler':
                options.sampler = args[++i];
                break;
            case '--url':
                options.url = args[++i];
                break;
            case '--img2img':
            case '-i':
                options.img2img = args[++i];
                break;
            case '--denoise':
                options.denoise = parseFloat(args[++i]);
                break;
            case '--help':
            case '-h':
                console.log(`
Stable Diffusion WebUI Automation - JavaScript Version

Usage: node automate_sd.js [options]

Options:
  -p, --prompt <text>     Text prompt (required)
  -n, --negative <text>   Negative prompt
  -o, --output <file>    (default: output Output filename.png)
  -s, --steps <num>      Number of steps (default: 20)
  --cfg <num>            CFG scale (default: 7.0)
  --width <num>          Image width (default: 512)
  --height <num>         Image height (default: 512)
  --seed <num>           Seed (-1 for random)
  --sampler <name>       Sampler (default: Euler a)
  --url <url>            WebUI URL (default: http://127.0.0.1:7860)
  -i, --img2img <file>   Input image for img2img
  --denoise <num>        Denoising strength (default: 0.75)
  -h, --help            Show this help

Examples:
  node automate_sd.js -p "a cat" -o cat.png
  node automate_sd.js --prompt "landscape" --steps 50 --width 1024 --height 576
  node automate_sd.js -i input.png --prompt "oil painting" --denoise 0.6
                `);
                process.exit(0);
        }
    }

    if (!options.prompt) {
        console.error('Error: --prompt is required');
        console.log('Use --help for usage information');
        process.exit(1);
    }

    const api = new StableDiffusionAPI(options.url);
    
    if (!await api.waitForApi()) {
        console.error('Error: Could not connect to WebUI');
        process.exit(1);
    }

    try {
        if (options.img2img) {
            // Image-to-image mode
            const imageB64 = api.readImageAsBase64(options.img2img);
            const images = await api.img2img({
                prompt: options.prompt,
                negative_prompt: options.negative,
                init_images: [imageB64],
                steps: options.steps,
                cfg_scale: options.cfg,
                denoising_strength: options.denoise,
                seed: options.seed,
                sampler_name: options.sampler
            });
            if (images.length > 0) {
                api.saveImage(images[0], options.output);
                console.log(`Saved: ${options.output}`);
            }
        } else {
            // Text-to-image mode
            await api.generate({
                prompt: options.prompt,
                negative_prompt: options.negative,
                output: options.output,
                steps: options.steps,
                cfg_scale: options.cfg,
                width: options.width,
                height: options.height,
                seed: options.seed,
                sampler_name: options.sampler
            });
        }
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

// Export for use as module
module.exports = { StableDiffusionAPI, apiRequest };

// Run CLI if executed directly
if (require.main === module) {
    main();
}
