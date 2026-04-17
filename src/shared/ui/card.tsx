import { cn } from "@/shared/lib";

type CardProps = React.HTMLAttributes<HTMLDivElement>;

export function Card({ className, ...props }: CardProps) {
  return (
    <div
      className={cn("rounded-2xl border border-white/10 bg-white/5 p-6", className)}
      {...props}
    />
  );
}
