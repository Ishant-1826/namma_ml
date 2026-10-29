import { defineConfig } from "vite";
import fs from "fs";
import path from "path";

export default defineConfig({
  base: "/",
  plugins: [
    {
      name: "clean-splats-from-dist",
      closeBundle() {
        const distScene = path.resolve(__dirname, "dist/scene");
        if (fs.existsSync(distScene)) {
          // Remove heavy uncompressed splat files (.ply and .splat) from dist
          // Keep scene.v1.spz (8.64 MB), which is well below the 25 MiB Pages per-file limit
          const files = fs.readdirSync(distScene);
          for (const file of files) {
            if (file.endsWith(".ply") || file.endsWith(".splat")) {
              fs.unlinkSync(path.join(distScene, file));
              console.log(`[clean-splats-from-dist] Excluded ${file} from Pages build.`);
            }
          }
        }
      },
    },
  ],
});
