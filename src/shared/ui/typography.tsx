import { cn } from "@/shared/lib";

type HeadingProps = React.HTMLAttributes<HTMLHeadingElement>;

export function H1({ className, ...props }: HeadingProps) {
  return (
    <h1 className={cn("text-4xl font-bold tracking-tight md:text-6xl", className)} {...props} />
  );
}

type TextProps = React.HTMLAttributes<HTMLParagraphElement>;

export function Text({ className, ...props }: TextProps) {
  return <p className={cn("text-white/70", className)} {...props} />;
}
