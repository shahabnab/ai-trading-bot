import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Trading Bot",
  description: "Personal AI-assisted trading dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
