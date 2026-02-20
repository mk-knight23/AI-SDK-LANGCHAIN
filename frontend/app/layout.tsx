export const metadata = {
  title: 'VentureGraph',
  description: 'Startup ecosystem intelligence platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
