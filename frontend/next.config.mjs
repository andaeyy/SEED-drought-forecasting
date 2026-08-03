/** @type {import('next').NextConfig} */
const apiProxyTarget = (process.env.API_PROXY_TARGET ?? "http://localhost:8000").replace(/\/$/, "");

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiProxyTarget}/api/:path*`
      }
    ];
  }
};

export default nextConfig;
