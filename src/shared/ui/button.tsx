import { cn } from "@/shared/lib";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
};

export function Button({ variant = "primary", className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "rounded-lg px-6 py-3 font-medium transition-colors",
        variant === "primary" && "bg-[var(--accent)] text-white hover:bg-[var(--accent)]/90",
        variant === "secondary" &&
          "border border-white/20 bg-transparent text-white hover:bg-white/10",
        className,
      )}
      {...props}
    />
  );
}
