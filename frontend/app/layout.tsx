import type { Metadata } from "next";
import AutoRefresh from "../components/AutoRefresh";
import "./globals.css";
import "./dashboard.css";

export const metadata: Metadata = {
  title: "AI Trading Bot · Paper Model Arena",
  description: "Forward paper-trading dashboard with isolated model performance and trade ledgers",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <AutoRefresh intervalMs={30_000} />
        {children}
      </body>
    </html>
  );
}
