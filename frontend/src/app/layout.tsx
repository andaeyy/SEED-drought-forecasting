import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SEED Drought Outlook",
  description: "Great Plains ET and soil-moisture endpoint outlooks from NLDAS forcings"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
